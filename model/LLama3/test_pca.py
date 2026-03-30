#!/usr/bin/env python
# coding: utf-8

import os
import sys
import argparse
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_dataset
from torch.amp import autocast
from transformers import AutoTokenizer, AutoModelForCausalLM

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
                    input_texts.append(f"Question: {choice} {target}\nAnswer:")
        return {"input_texts": input_texts}
    return collate_fn

# ================= 绘图逻辑 =================
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

# ================= 主函数 =================
def main():
    parser = argparse.ArgumentParser(description="Test and Visualize Llama3 PCA Components")
    parser.add_argument("--dataset", type=str, required=True, choices=list(TASK_CONFIGS.keys()), help="Target dataset")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B", help="HuggingFace model name")
    parser.add_argument("--pca_path", type=str, required=True, help="Path to the saved PCA .pt file")
    parser.add_argument("--layer_id", type=int, default=0, help="Layer index to visualize")
    parser.add_argument("--target_type", type=str, default="value", choices=["key", "value"], help="Which projection to test (key or value)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for dataloader")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for dataloader")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_plot_path", type=str, default=None, help="Path to save the generated plot (.png)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    task_config = TASK_CONFIGS[args.dataset]

    print(f"Loading model {args.model_name} to {args.device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    num_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // model.config.num_attention_heads

    print(f"Loading PCA matrices from {args.pca_path}...")
    pca_outputs = torch.load(args.pca_path, map_location="cpu")
    
    target_matrices = pca_outputs[args.target_type][args.layer_id]
    if isinstance(target_matrices, list):
        P_matrix = torch.stack(target_matrices, dim=0)
    else:
        P_matrix = target_matrices
    # P_matrix shape: [num_kv_heads, head_dim, head_dim]
    P_matrix = P_matrix.to(torch.float32)

    print(f"Loading dataset: {task_config['path']}...")
    dataset = load_dataset(*task_config["path"].split(":"))
    test_loader = torch.utils.data.DataLoader(
        dataset["train"], 
        shuffle=False, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        pin_memory=True, 
        collate_fn=get_collate_fn(args.dataset)
    )

    sq_sum_list = []

    print(f"Running forward pass for Layer {args.layer_id} ({args.target_type})...")
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Aggregating L2 Norms"):
            input_texts = data["input_texts"]
            
            inputs = tokenizer(
                input_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(args.device)

            with autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(**inputs, use_cache=True)
            
            past_key_values = outputs.past_key_values
            
            if hasattr(past_key_values, "key_cache"):
                state = past_key_values.key_cache[args.layer_id] if args.target_type == "key" else past_key_values.value_cache[args.layer_id]
            else:
                state = past_key_values[args.layer_id][0 if args.target_type == "key" else 1]

            # state shape: [batch_size, num_kv_heads, seq_len, head_dim]
            # 转换为: [batch_size, seq_len, num_kv_heads, head_dim]
            state = state.transpose(1, 2)

            # # 带 Attention mask 实现：新
            # attention_mask = inputs["attention_mask"].bool().flatten()
            # flat_state = state.reshape(-1, num_kv_heads, head_dim)
            # valid_state = flat_state[attention_mask].to(torch.float32)

            # 不带 Attention mask 实现：旧
            flat_state = state.reshape(-1, num_kv_heads, head_dim)
            valid_state = flat_state.to(torch.float32)

            # 使用 einsum 批量执行矩阵乘法
            # valid_state: [T, H, D], P_matrix: [H, D, P] -> rotated: [T, H, P]
            rotated = torch.einsum('thd,hdp->thp', valid_state, P_matrix.to(args.device))
            
            # 累加当前 batch 的平方和 (维度 0 是 valid_tokens)
            sq_sum = torch.sum(rotated ** 2, dim=0).detach().cpu()  # [num_kv_heads, head_dim]
            sq_sum_list.append(sq_sum)

            del outputs, past_key_values, inputs, state, flat_state, valid_state, rotated

    # 全局平方和累加开方，获得精确的 L2 范数
    total_sq_sum = torch.stack(sq_sum_list).sum(dim=0)
    global_l2_results = torch.sqrt(total_sq_sum).numpy()
    
    # 按注意力头进行 Min-Max 归一化
    norm_l2_results_by_head = []
    for h in range(num_kv_heads):
        head_l2 = global_l2_results[h]
        norm_head_l2 = (head_l2 - head_l2.min()) / (head_l2.max() - head_l2.min() + 1e-8)
        norm_l2_results_by_head.append(norm_head_l2)
        
    print("Plotting results...")
    plot_title_prefix = f"Llama-3 Layer {args.layer_id} L2 Norms ({args.target_type.capitalize()})"
    plot_pca_norms_subplots(norm_l2_results_by_head, title_prefix=plot_title_prefix, save_path=args.save_plot_path)

if __name__ == "__main__":
    main()