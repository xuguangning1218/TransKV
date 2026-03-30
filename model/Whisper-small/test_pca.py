#!/usr/bin/env python
# coding: utf-8

import os
import sys
import argparse
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_dataset, Audio
from dataclasses import dataclass
from typing import Any, Dict, List
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers import logging
logging.set_verbosity_error()  # 屏蔽 Fast Tokenizer 的警告

# 动态获取当前脚本所在目录的上一级目录并加入环境变量
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from utils import set_seed

# ================= 严格对齐提取时的数据加载逻辑 =================
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_features = [feature["input_features"] for feature in features]
        audio_lengths = [len(x) for x in input_features]

        padded_features = self.processor.feature_extractor.pad(
            {"input_features": input_features}, return_tensors="pt"
        )
        input_features_pt = padded_features["input_features"]

        max_len = input_features_pt.shape[-1]
        attention_mask = torch.zeros((len(input_features), max_len), dtype=torch.long)
        for i, length in enumerate(audio_lengths):
            attention_mask[i, :length] = 1

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
        
        if sample_ratio < 1.0:
            dataset = dataset.shuffle(seed=seed).select(range(int(len(dataset) * sample_ratio)))

        def prepare_dataset(batch):
            audio = batch["audio"]
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
        # 为了严格复现PCA分布，这里是否shuffle都可以，因为我们会跑全量
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True, 
        collate_fn=collate_fn
    )
    return loader

# ================= 绘图与 Hook 逻辑 =================
def plot_pca_norms_subplots(norms_by_head, title_prefix="PCA Component Norms", save_path=None):
    num_heads = len(norms_by_head)
    head_dim = len(norms_by_head[0])
    
    num_cols = int(np.ceil(np.sqrt(num_heads)))
    num_rows = int(np.ceil(num_heads / num_cols))
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 4), sharey=True)
    fig.suptitle(f"{title_prefix} - By Head", fontsize=16)

    max_power = int(np.log2(head_dim))
    tick_labels = [2**i for i in range(1, max_power + 1)]
    tick_positions = [val - 1 for val in tick_labels]

    for h in range(num_heads):
        norms_np = norms_by_head[h]
        x = np.arange(len(norms_np))
        width = 0.5
        
        row = h // num_cols
        col = h % num_cols
        ax = axes[row, col] if num_rows > 1 else axes[col]
        
        ax.bar(x, norms_np, width, color='skyblue', alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_title(f"Head {h + 1}", fontsize=12)
        ax.set_xlabel("Component Index", fontsize=10)
        ax.set_ylabel("Normalized L2 Norm", fontsize=10)
        
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.grid(True, axis='y', alpha=0.3)

    for i in range(num_heads, num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        ax = axes[row, col] if num_rows > 1 else axes[col]
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])  
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def insert_encoder_v_hooks(model, layer_id, P_matrix):
    value_hooks = []
    # 存储每个 batch 算出来的 [num_heads, head_dim] 的平方和
    sq_sum_list = []

    config = model.config
    num_heads = config.encoder_attention_heads
    head_dim = config.d_model // num_heads

    def value_hook_fn(module, input, output):
        item = output.detach()
        bsz, seq_len, _ = item.shape
        item_reshaped = item.view(bsz, seq_len, num_heads, head_dim)
        
        item_reshaped_rotateds = []
        for g in range(num_heads):
            rotated = torch.matmul(item_reshaped[:, :, g], P_matrix[g].to(item.device))
            item_reshaped_rotateds.append(rotated)
            
        item_reshaped_rotateds = torch.stack(item_reshaped_rotateds, dim=2)
        flat = item_reshaped_rotateds.view(-1, num_heads, head_dim)
        
        # 为了内存安全和全局 L2 范数准确性，我们在这个阶段先累加特征的平方和
        sq_sum = torch.sum(flat ** 2, dim=0).detach()  # Shape: [num_heads, head_dim]
        sq_sum_list.append(sq_sum)

    # Whisper Encoder 的 v_proj
    attn_module = model.model.encoder.layers[layer_id].self_attn
    value_hook = attn_module.v_proj.register_forward_hook(value_hook_fn)
    value_hooks.append(value_hook)

    return value_hooks, sq_sum_list

# ================= 主函数 =================
def main():
    parser = argparse.ArgumentParser(description="Test and Visualize Whisper Encoder PCA Components")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--model_name", type=str, default="discoverylabs/whisper-small-english-finetuned", help="HuggingFace model ID")
    parser.add_argument("--data_path", type=str, default="/data/CLOVER-main/data/", help="Directory to cache dataset")
    parser.add_argument("--pca_path", type=str, required=True, help="Path to the saved Whisper PCA .pt file")
    parser.add_argument("--layer_id", type=int, default=0, help="Encoder layer index to visualize")
    parser.add_argument("--sample_ratio", type=float, default=0.05, help="Must match extract_pca_whisper.py exactly")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--seed", type=int, default=42, help="Must match extract_pca_whisper.py exactly")
    parser.add_argument("--save_plot_path", type=str, default=None, help="Path to save the generated plot")
    args = parser.parse_args()

    set_seed(args.seed)

    print(f"Loading model {args.model_name} to {args.device}...")
    processor = WhisperProcessor.from_pretrained(args.model_name)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
    
    # 保持配置一致
    model.generation_config.forced_decoder_ids = None
    model.generation_config.begin_suppress_tokens = []  
    model.generation_config.language = "english"
    model.generation_config.task = "transcribe"
    model.generation_config.no_timestamps = True
    
    model.to(args.device)
    model.eval()

    print(f"Loading PCA matrices from {args.pca_path}...")
    pca_outputs = torch.load(args.pca_path, map_location="cpu")
    
    # 精确获取 encoder_self 的 value 投影矩阵
    pca_encoder_value = pca_outputs['encoder']['value']
    
    if isinstance(pca_encoder_value[args.layer_id], list):
        P = torch.stack(pca_encoder_value[args.layer_id], dim=0)
    else:
        P = pca_encoder_value[args.layer_id]

    print("Initializing DataLoader (Strictly matching extraction data)...")
    dataloader = get_dataloader(processor, args.batch_size, args.num_workers, args.sample_ratio, args.seed, args.data_path)

    print(f"Registering hooks for Encoder Layer {args.layer_id}...")
    value_hooks, sq_sum_list = insert_encoder_v_hooks(model, args.layer_id, P)

    print("Running forward pass through the FULL sampled dataset...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Aggregating L2 Norms"):
            input_features = batch["input_features"].to(args.device)
            labels = batch["labels"].to(args.device)
            model(input_features=input_features, labels=labels)
            break

    for hook in value_hooks:
        hook.remove()

    # 将所有 batch 的平方和累加，再开方，得到全局准确的 L2 范数
    total_sq_sum = torch.stack(sq_sum_list).sum(dim=0)  # [num_heads, head_dim]
    global_l2_results = torch.sqrt(total_sq_sum).cpu().numpy()
    
    # 每个头独立做 Min-Max 归一化
    norm_l2_results_by_head = []
    for h in range(len(global_l2_results)):
        head_l2 = global_l2_results[h]
        norm_head_l2 = (head_l2 - head_l2.min()) / (head_l2.max() - head_l2.min() + 1e-8)
        norm_l2_results_by_head.append(norm_head_l2)
        
    print("Plotting results...")
    plot_title_prefix = f"Whisper Encoder Layer {args.layer_id} L2 Norms (Value)"
    plot_pca_norms_subplots(norm_l2_results_by_head, title_prefix=plot_title_prefix, save_path=args.save_plot_path)

if __name__ == "__main__":
    main()