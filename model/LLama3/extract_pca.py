#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import re
import torch
import torch.nn as nn
from tqdm import tqdm
from datasets import load_dataset
from torch.amp import autocast
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import pca_calc, set_seed

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ================= 任务配置与映射 =================
TASK_CONFIGS = {
    "arc_challenge":{"path": "allenai/ai2_arc:ARC-Challenge",    "display": "ARC-Challenge"},
    "arc_easy":     {"path": "allenai/ai2_arc:ARC-Easy",         "display": "ARC-Easy"},
    "boolq":        {"path": "super_glue:boolq",                 "display": "BoolQ"},
    "hellaswag":    {"path": "Rowan/hellaswag",                  "display": "HellaSwag"},
    "openbookqa":   {"path": "allenai/openbookqa",               "display": "OpenBookQA"},
    "piqa":         {"path": "baber/piqa",                       "display": "PIQA"},
    "winogrande":   {"path": "allenai/winogrande:winogrande_xl", "display": "WinoGrande"},
}

# ================= 数据处理模块 =================
def get_collate_fn(task_name):
    """根据不同的任务生成对应的 collate_fn，统一返回格式化的 input_texts"""
    
    def collate_fn(batch):
        input_texts = []
        
        if task_name in ["arc_challenge", "arc_easy"]:
            for item in batch:
                choices = item["choices"]
                c_text = " ".join(choices["text"]) if isinstance(choices, dict) and "text" in choices else " ".join(choices)
                input_texts.append(f"Question:{item['question']}\nAnswer:{c_text}")
                
        elif task_name == "boolq":
            for item in batch:
                input_texts.append(f"{item['passage']}\nQuestion: {item['question']}?\nAnswer: [no, yes]")
                
        elif task_name == "hellaswag":
            for item in batch:
                ctx = item["ctx_a"] + " " + item["ctx_b"].capitalize()
                query = item["activity_label"] + ": " + ctx
                # HellaSwag 特定的清洗逻辑
                query = query.strip().replace(" [title]", ". ")
                query = re.sub("\\[.*?\\]", "", query).replace("  ", " ")
                input_texts.append(query)
                
        elif task_name == "openbookqa":
            for item in batch:
                choices = item["choices"]
                c_text = " ".join(choices["text"]) if isinstance(choices, dict) and "text" in choices else " ".join(choices)
                input_texts.append(f"{item['question_stem']} {c_text}")
                
        elif task_name == "piqa":
            for item in batch:
                input_texts.append(f"Question: {item['goal']}\nAnswer:[{item['sol1']}, {item['sol2']}]")
                
        elif task_name == "winogrande":
            for item in batch:
                idx = item["sentence"].index("_")
                target = item["sentence"][idx + 1:].strip()
                options = [item["option1"], item["option2"]]
                choices = [item["sentence"][:idx] + opt for opt in options]
                for choice in choices:
                    # 将一个样本展开为两个完整的输入进行校准
                    input_texts.append(f"Question: {choice} {target}\nAnswer:")
                    
        return {"input_texts": input_texts}
    
    return collate_fn

# ================= 核心提取流程 =================
def get_kv_cache_for_pca(model, tokenizer, trainloader, save_path, layers_per_chunk=8):
    """
    layers_per_chunk: 每次前向传播处理的层数，可有效控制 CPU OOM 问题。
    """
    pca_value_outputs = []
    pca_key_outputs = []
    device = model.device
    num_layers = len(model.model.layers)
    num_key_value_heads = model.model.config.num_key_value_heads

    model.eval()

    # 预先计算分块的层数范围
    chunk_ranges = []
    for i in range(0, num_layers, layers_per_chunk):
        chunk_ranges.append((i, min(i + layers_per_chunk, num_layers)))

    with torch.no_grad():
        for chunk_start, chunk_end in chunk_ranges:
            print(f"\n=== Processing Layer Chunk: [{chunk_start} to {chunk_end - 1}] ===")
            
            # 为当前 Chunk 分配 CPU 暂存字典
            chunk_layer_keys_cpu = {l: [] for l in range(chunk_start, chunk_end)}
            chunk_layer_values_cpu = {l: [] for l in range(chunk_start, chunk_end)}

            for data in tqdm(trainloader, desc=f"Forward Pass (Layers {chunk_start}-{chunk_end-1})"):
                input_texts = data["input_texts"]
                
                inputs = tokenizer(
                    input_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    # max_length=4096,
                ).to(device)

                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs = model(**inputs, use_cache=True)
                
                past_key_values = outputs.past_key_values
                attention_mask = inputs["attention_mask"].bool()
                
                for layer_id in range(chunk_start, chunk_end):
                    if hasattr(past_key_values, "key_cache"):
                        k_state = past_key_values.key_cache[layer_id]
                        v_state = past_key_values.value_cache[layer_id]
                    else:
                        k_state = past_key_values[layer_id][0]
                        v_state = past_key_values[layer_id][1]
                    
                    # 1. 调整维度为 [batch_size, seq_len, num_kv_heads, head_dim]
                    k_state = k_state.transpose(1, 2)
                    v_state = v_state.transpose(1, 2)
                    
                    # # 带 Attention mask 实现：新
                    # layer_device = k_state.device
                    # layer_attention_mask = attention_mask.to(layer_device)
                    # valid_k_state = k_state[layer_attention_mask]
                    # valid_v_state = v_state[layer_attention_mask]
                    # valid_k_state_4d = valid_k_state.unsqueeze(0).to('cpu', dtype=torch.float32, non_blocking=True)
                    # valid_v_state_4d = valid_v_state.unsqueeze(0).to('cpu', dtype=torch.float32, non_blocking=True)

                    # 不带 Attention mask 实现：旧
                    valid_k_state_4d = k_state.to('cpu', dtype=torch.float32, non_blocking=True)
                    valid_v_state_4d = v_state.to('cpu', dtype=torch.float32, non_blocking=True)
                    
                    chunk_layer_keys_cpu[layer_id].append(valid_k_state_4d)
                    chunk_layer_values_cpu[layer_id].append(valid_v_state_4d)

                del outputs, past_key_values, inputs, attention_mask

            print(f"Calculating PCA for layers {chunk_start} to {chunk_end-1}...")
            for layer_id in tqdm(range(chunk_start, chunk_end), desc="PCA Calculation"):
                # 直接将暂存了伪 4D 张量的列表传递给 pca_calc，防止全局拼接导致内存爆炸
                pca_key_output = pca_calc(chunk_layer_keys_cpu[layer_id], device, num_key_value_heads)
                pca_key_outputs.append(pca_key_output)

                pca_value_output = pca_calc(chunk_layer_values_cpu[layer_id], device, num_key_value_heads)
                pca_value_outputs.append(pca_value_output)

                # 及时释放内存
                del chunk_layer_keys_cpu[layer_id]
                del chunk_layer_values_cpu[layer_id]
                
    pca_outputs = {
        "key": pca_key_outputs,
        "value": pca_value_outputs,
    }

    torch.save(pca_outputs, save_path)
    print(f"\nSuccessfully saved PCA rotation matrices to {save_path}")
    return pca_outputs

# ================= 主函数 =================
def main():
    parser = argparse.ArgumentParser(description="Llama3 PCA Constructor for Various Datasets")
    parser.add_argument("--dataset", type=str, required=True, choices=list(TASK_CONFIGS.keys()), help="Target dataset for PCA extraction")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B", help="HuggingFace model name")
    parser.add_argument("--save_dir", type=str, default=f"/data/TransKV/pca_matrices/LLama/", help="Directory to save PCA pt files")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for dataloader")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for dataloader")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--layers_per_chunk", type=int, default=32, help="Number of layers to process in each forward pass chunk")
    args = parser.parse_args()

    # 基础设置
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    
    task_config = TASK_CONFIGS[args.dataset]
    dataset_path = task_config["path"]
    display_name = task_config["display"]
    pca_save_path = os.path.join(args.save_dir, f"{args.dataset}.pt")

    print(f"Handling {display_name} dataset...")

    # 模型加载
    print(f"Loading model {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, device_map="auto")

    # 数据集加载
    print(f"Loading dataset: {dataset_path}...")
    dataset_parts = dataset_path.split(":")
    dataset = load_dataset(*dataset_parts)
    train_dataset = dataset["train"]
    print(f"Train dataset length: {len(train_dataset)}")

    # 初始化 DataLoader
    collate_fn = get_collate_fn(args.dataset)
    trainloader = torch.utils.data.DataLoader(
        train_dataset, 
        shuffle=False, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        pin_memory=True, 
        collate_fn=collate_fn
    )

    # 提取 PCA
    print("Starting PCA extraction process...")
    get_kv_cache_for_pca(model, tokenizer, trainloader, pca_save_path, layers_per_chunk=args.layers_per_chunk)

if __name__ == "__main__":
    main()