#!/usr/bin/env python
# coding: utf-8

import os
import sys
import argparse
import pickle
import torch
from tqdm import tqdm
from datasets import load_dataset, Audio
from dataclasses import dataclass
from typing import Any, Dict, List
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# 动态获取当前脚本所在目录的上一级目录并加入环境变量
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from utils import set_seed, pca_calc

# ================= 数据收集器 =================
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # 提取原始音频
        input_features = [feature["input_features"] for feature in features]
        audio_lengths = [len(x) for x in input_features]

        # padding
        padded_features = self.processor.feature_extractor.pad(
            {"input_features": input_features}, return_tensors="pt"
        )
        input_features_pt = padded_features["input_features"]

        # 构造 attention_mask
        max_len = input_features_pt.shape[-1]
        attention_mask = torch.zeros((len(input_features), max_len), dtype=torch.long)
        for i, length in enumerate(audio_lengths):
            attention_mask[i, :length] = 1

        # 标签处理
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]

        return {
            "input_features": input_features_pt,
            "attention_mask": attention_mask,
            "labels": labels,
        }

# ================= 数据加载器 =================
def get_dataloader(processor, batch_size, num_workers, sample_ratio, seed, data_path):
    language = "en"
    split = "train"
    full_path = os.path.join(data_path, f"common_voice_17_0_{language}_{split}_16khz_sampled.pkl")

    if os.path.exists(full_path):
        print(f"Loading cached dataset from {full_path}...")
        with open(full_path, 'rb') as f:
            dataset = pickle.load(f)
    else:
        print(f"Preparing dataset for {language}...")
        dataset = load_dataset("mozilla-foundation/common_voice_17_0", language, split=split, cache_dir=data_path)
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
        
        # 按照采样率进行采样
        if sample_ratio < 1.0:
            dataset = dataset.shuffle(seed=seed).select(range(int(len(dataset) * sample_ratio)))

        def prepare_dataset(batch):
            audio = batch["audio"]
            # Whisper processor 期望输入 16kHz 的一维数组
            batch["input_features"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
            batch["labels"] = processor.tokenizer(batch["sentence"], truncation=True).input_ids
            return batch

        dataset = dataset.map(prepare_dataset, num_proc=num_workers, remove_columns=dataset.column_names)
        
        # with open(full_path, 'wb') as f:
        #     pickle.dump(dataset, f)

    collate_fn = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True, 
        collate_fn=collate_fn
    )
    return loader

# ================= Hook 注册 =================
def insert_hooks(attention_module, layer_idx, target_types, num_heads, head_dim, outputs_dict):
    hooks = []
    
    def make_hook_fn(target_type):
        def hook_fn(module, input, output):
            if layer_idx not in outputs_dict[target_type]:
                outputs_dict[target_type][layer_idx] = []
            
            # Whisper 的 q/k/v/out_proj 输出形状为 [batch_size, seq_len, embed_dim]
            # 为了计算各头的 PCA，需要 reshape 为 [batch_size, seq_len, num_heads, head_dim]
            bsz, seq_len, _ = output.shape
            reshaped_output = output.view(bsz, seq_len, num_heads, head_dim).to('cpu')
            outputs_dict[target_type][layer_idx].append(reshaped_output)
        return hook_fn

    if "query" in target_types:
        hooks.append(attention_module.q_proj.register_forward_hook(make_hook_fn("query")))
    if "key" in target_types:
        hooks.append(attention_module.k_proj.register_forward_hook(make_hook_fn("key")))
    if "value" in target_types:
        hooks.append(attention_module.v_proj.register_forward_hook(make_hook_fn("value")))
    if "output" in target_types:
        hooks.append(attention_module.out_proj.register_forward_hook(make_hook_fn("output")))

    return hooks

# ================= 核心提取流程 =================
def process_attention_blocks(model, dataloader, device, block_type, module_list, target_types):
    """
    block_type: 'encoder_self', 'decoder_self', 或 'decoder_cross'
    module_list: 对应的 nn.ModuleList (例如 model.model.encoder.layers)
    """
    pca_results = {t: [] for t in target_types}
    config = model.config
    
    # 动态获取 heads 和 dim
    num_heads = config.encoder_attention_heads if 'encoder' in block_type else config.decoder_attention_heads
    head_dim = config.d_model // num_heads

    for layer_id, layer_module in enumerate(module_list):
        print(f"Processing {block_type} layer {layer_id} for {target_types}...")
        
        # 根据 block_type 获取具体的 attention 模块
        if block_type == 'encoder_self' or block_type == 'decoder_self':
            attn_module = layer_module.self_attn
        elif block_type == 'decoder_cross':
            attn_module = layer_module.encoder_attn

        outputs_dict = {t: {} for t in target_types}
        hooks = insert_hooks(attn_module, layer_id, target_types, num_heads, head_dim, outputs_dict)

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Layer {layer_id} Forward"):
                input_features = batch["input_features"].to(device)
                labels = batch["labels"].to(device)
                
                # 使用 labels 进行 Teacher Forcing 前向传播，能够一次性提取 Decoder 所有 step 的激活值
                model(input_features=input_features, labels=labels)
                torch.cuda.empty_cache()

        for hook in hooks:
            hook.remove()

        # 计算并保存当前层的 PCA
        for t in target_types:
            pca_res = pca_calc(outputs_dict[t][layer_id], device, num_heads)
            pca_results[t].append(pca_res)
            del outputs_dict[t][layer_id]
        
        torch.cuda.empty_cache()

    return pca_results

def main():
    parser = argparse.ArgumentParser(description="Whisper PCA Constructor")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--model_name", type=str, default="discoverylabs/whisper-small-english-finetuned", help="HuggingFace model ID")
    parser.add_argument("--data_path", type=str, default="/data/CLOVER-main/data/CommonVoice_16khz/", help="Directory to cache dataset")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save PCA pt files")
    parser.add_argument("--sample_ratio", type=float, default=0.05, help="Ratio of dataset to sample")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for dataloader")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for dataloader")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.data_path, exist_ok=True)

    if args.sample_ratio < 1:
        pca_save_path = os.path.join(args.save_dir, f"CommonVoice_English_{int(args.sample_ratio * 100)}.pt")
    else:
        pca_save_path = os.path.join(args.save_dir, 'CommonVoice_English.pt')
    
    print(f"Loading model {args.model_name} to {args.device}...")
    processor = WhisperProcessor.from_pretrained(args.model_name)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
    
    # 按照你的要求配置 Generation Config
    model.generation_config.forced_decoder_ids = None
    model.generation_config.begin_suppress_tokens = []  
    model.generation_config.language = "english"
    model.generation_config.task = "transcribe"
    model.generation_config.no_timestamps = True
    
    model.to(args.device)
    model.eval()

    print("Initializing DataLoader...")
    dataloader = get_dataloader(processor, args.batch_size, args.num_workers, args.sample_ratio, args.seed, args.data_path)

    target_types = ["key", "value"]
    
    # 分别处理 Encoder 和 Decoder 的 Attention 模块
    print("\n=== Starting Encoder Self-Attention Phase ===")
    encoder_self_results = process_attention_blocks(
        model, dataloader, args.device, 'encoder_self', model.model.encoder.layers, target_types
    )

    print("\n=== Starting Decoder Self-Attention Phase ===")
    decoder_self_results = process_attention_blocks(
        model, dataloader, args.device, 'decoder_self', model.model.decoder.layers, target_types
    )

    print("\n=== Starting Decoder Cross-Attention Phase ===")
    decoder_cross_results = process_attention_blocks(
        model, dataloader, args.device, 'decoder_cross', model.model.decoder.layers, target_types
    )

    print("\n=== Combining and Saving Results ===")
    combined_results = {
        "encoder": encoder_self_results,
        "decoder": decoder_self_results,
        "cross": decoder_cross_results
    }
    
    torch.save(combined_results, pca_save_path)
    print(f"Successfully saved Whisper PCA rotation matrices to {pca_save_path}")

if __name__ == "__main__":
    main()