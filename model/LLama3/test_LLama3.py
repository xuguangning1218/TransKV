#!/usr/bin/env python
# coding: utf-8

import os
import sys
import json
import datetime
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from lm_eval import simple_evaluate
from lm_eval.utils import handle_non_serializable
from utils import compute_pruned_dim
from transkv import transkv_llama3

# ==================== 文件保存模块 ====================
def llama3_files_saver(model, tokenizer, save_directory, args):
    """
    Save the Llama3 model to the specified directory.
    """
    print(f"Saving model and tokenizer to {save_directory}...")
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    
    # 获取原始 config 字典并添加新字段
    config_dict = model.config.to_dict()
    config_dict["pruned_dim"] = args.pruned_dim
    config_dict["head_dim_qk"] = model.config.hidden_size // model.config.num_attention_heads
    config_dict["pca_save_path"] = args.pca_path
    
    # 手动保存 config.json
    with open(f"{save_directory}/config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
    print(f"Updated config.json with pruned_dim={args.pruned_dim}")

# ==================== 模型评估模块 ====================
def llama3_evaluation(model_name: str, tasks: str, output_dir: str):
    """
    Evaluate the model using lm_eval
    """
    print(f"Starting evaluation on tasks: {tasks} for model: {model_name}")
    results = simple_evaluate(
        model="hf",  # 指定模型类型为 Hugging Face
        model_args=f"pretrained={model_name}",  # 指定刚刚保存的模型路径
        tasks=tasks, # 任务列表（逗号分隔字符串）
        batch_size="auto",  # 批处理大小自动推断
    )

    now = datetime.datetime.now()
    timestamp = now.isoformat().replace(":", "-")  # 生成时间戳
    output_file = f"{output_dir}/results_{timestamp}.json" 

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(
            results, f, indent=2, default=handle_non_serializable, ensure_ascii=False
        )
        
    print(f"Evaluation completed. Results saved to: {output_file}")
    return results

# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser(description="Prune, Save and Evaluate Llama3 Model")
    
    # 基础模型与设备
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B", help="Original HuggingFace model ID")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to load the original model")
    
    # 剪枝配置
    parser.add_argument("--pca_path", type=str, default=None, help="Path to PCA .pt file. If None, evaluates original model.")
    parser.add_argument("--pruned_ratio", type=float, default=0.5, help="Ratio of dimensions to prune")
    
    # 保存与评估配置
    parser.add_argument("--pruned_output_dir", type=str, default="/data/TransKV/prunned_saved_model/Llama3/", help="Base directory for saved models")
    parser.add_argument("--tasks", type=str, required=True, choices=['arc_challenge', 'arc_easy', 'boolq', 'hellaswag', 'openbookqa', 'piqa', 'winogrande'], help="Comma-separated list of lm_eval tasks")
    parser.add_argument("--eval_output_dir", type=str, default="/data/TransKV/results/Llama3/", help="Directory to save evaluation json results")
    
    args = parser.parse_args()

    # 加载原始模型与 Tokenizer
    print(f"Loading original model {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # Llama3 默认没有 pad_token，设置 EOS 作为 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )

    
    args.pruned_dim = compute_pruned_dim(model.model, args.pruned_ratio)

    target_model_path = args.model_name # 默认评估原模型路径

    # 模型剪枝
    if args.pca_path is not None and args.pca_path.strip() != "":
        print(f"PCA matrices found at: {args.pca_path}. Pruning model...")
        model = transkv_llama3(model, args)
        
        # 设定保存目录
        pruned_model_dir = os.path.join(args.pruned_output_dir, f"pruned-transkv-{int(args.pruned_ratio*100)}")
        os.makedirs(pruned_model_dir, exist_ok=True)
        
        # 保存剪枝模型
        llama3_files_saver(model, tokenizer, pruned_model_dir, args)
        
        # 将后续评估的路径指向保存的本地剪枝模型目录
        target_model_path = pruned_model_dir
    else:
        print("No PCA path provided. Evaluating original model without pruning.")

    # 释放显存
    print("Clearing GPU memory before lm_eval evaluation...")
    del model
    del tokenizer
    torch.cuda.empty_cache()

    # 调用 lm_eval 执行任务测试
    llama3_evaluation(
        model_name=target_model_path,
        tasks=args.tasks,
        output_dir=args.eval_output_dir
    )

if __name__ == "__main__":
    main()