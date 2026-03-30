#!/usr/bin/env python
# coding: utf-8

import os
import sys
import argparse
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from transformers import ViTForImageClassification

# 动态获取当前脚本所在目录的上一级目录（即 /data/TransKV/），并加入环境变量
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from utils import set_seed

def plot_pca_norms_subplots(norms_by_head, title_prefix="PCA Component Norms", save_path=None):
    """
    绘制并保存多个子图，每个子图展示一个 attention head 的 PCA 主成分 L2 范数。
    
    norms_by_head: 形状为 (num_heads, head_dim) 的 NumPy 数组
    """
    num_heads = len(norms_by_head)
    
    # 计算子图网格的行数和列数（例如，尽可能接近正方形）
    num_cols = int(np.ceil(np.sqrt(num_heads)))
    num_rows = int(np.ceil(num_heads / num_cols))
    
    # 创建具有指定网格大小的 Figure 和 Axes
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 6, num_rows * 5), sharey=True)
    fig.suptitle(f"{title_prefix} - By Head", fontsize=16)

    for h in range(num_heads):
        # 获取第 h 个头的归一化范数数据
        norms_np = norms_by_head[h].cpu().numpy() if torch.is_tensor(norms_by_head[h]) else norms_by_head[h]
        
        x = np.arange(len(norms_np))
        width = 0.5
        
        # 确定当前头所在的行和列
        row = h // num_cols
        col = h % num_cols
        
        # 选择对应的 Axes 对象
        ax = axes[row, col] if num_rows > 1 else axes[col]
        
        # 绘制柱状图
        ax.bar(x, norms_np, width, color='skyblue', alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # 设置子图的细节
        ax.set_title(f"Head {h + 1}", fontsize=12)
        ax.set_xlabel("Component Index", fontsize=10)
        ax.set_ylabel("Normalized L2 Norm", fontsize=10)
        
        step = max(1, len(norms_np) // 6) 
        tick_positions = np.arange(0, len(norms_np), step)
        
        # 强制包含最后一个特征索引（例如 64）
        if len(norms_np) - 1 not in tick_positions:
            tick_positions = np.append(tick_positions, len(norms_np) - 1)
            
        tick_labels = tick_positions + 1  # 刻度标签从 1 开始
        
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        # 根据 head_dim 的大小，考虑是否需要旋转 x 轴刻度标签以避免重叠
        # ax.tick_params(axis='x', labelrotation=90)
        ax.grid(True, axis='y', alpha=0.3)

    # 自动调整布局，防止重叠
    plt.tight_layout(rect=[0, 0, 1, 0.96])  
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def insert_v_hooks_for_test(model, layer_id, P_matrix):
    """
    插入 Hook 计算经过 PCA 投影后的 Value 特征的 L2 范数
    P_matrix shape: [num_heads, head_dim, head_dim]
    """
    value_hooks = []
    value_outputs = []

    config = model.config
    num_heads = config.num_attention_heads
    head_dim = config.hidden_size // num_heads

    def value_hook_fn(module, input, output):
        # output shape: [batch_size, seq_len, hidden_size]
        item = output.detach()
        
        # Reshape to [batch, seq, num_heads, head_dim]
        item_reshaped = item.reshape(item.shape[0], item.shape[1], num_heads, head_dim)
        
        item_reshaped_rotateds = []
        for g in range(num_heads):
            # 将每个头的特征与其对应的 PCA 投影矩阵相乘
            rotated = torch.matmul(item_reshaped[:, :, g], P_matrix[g].to(item.device))
            item_reshaped_rotateds.append(rotated)
            
        # 拼接所有头的结果: [batch, seq, num_heads * head_dim]
        item_reshaped_rotateds = torch.concatenate(item_reshaped_rotateds, dim=2)
        
        # 展平以便计算每个特征维度的全局 L2 范数: [-1, num_heads, head_dim]
        flat = item_reshaped_rotateds.reshape(-1, num_heads, head_dim)
        
        # 计算 L2 范数
        l2_results = torch.norm(flat, dim=0).detach().cpu().numpy()
        value_outputs.append(l2_results)

    self_attn = model.vit.encoder.layer[layer_id].attention
    if hasattr(self_attn.attention, "value"):
        value_hook = self_attn.attention.value.register_forward_hook(value_hook_fn)
        value_hooks.append(value_hook)

    return value_hooks, value_outputs

def main():
    parser = argparse.ArgumentParser(description="Test and Visualize PCA Components on ImageNet/CIFAR10")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--model_name", type=str, default="google/vit-base-patch16-224", help="HuggingFace model name")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset (ImageNet)")
    parser.add_argument("--pca_path", type=str, required=True, help="Path to the saved PCA .pt file")
    parser.add_argument("--layer_id", type=int, default=0, help="Layer index to visualize")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for a single forward pass")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_plot_path", type=str, default=None, help="Path to save the generated plot (.png)")
    args = parser.parse_args()

    set_seed(args.seed)

    print(f"Loading model {args.model_name} to {args.device}...")
    model = ViTForImageClassification.from_pretrained(args.model_name)
    model.to(args.device)
    model.eval()

    print(f"Loading PCA matrices from {args.pca_path}...")
    pca_qkvo_outputs = torch.load(args.pca_path, map_location="cpu")
    pca_value_outputs = pca_qkvo_outputs['value']
    
    # 获取指定层的 P 矩阵并堆叠为 [num_heads, head_dim, head_dim]
    if isinstance(pca_value_outputs[args.layer_id], list):
        P = torch.stack(pca_value_outputs[args.layer_id], dim=0)
    else:
        P = pca_value_outputs[args.layer_id]

    print("Initializing DataLoader for 1 batch forward pass...")
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 此处默认使用 ImageNet 作为验证数据以与你的代码逻辑对齐。
    # 如果原意是使用 CIFAR-10，可将此处替换为 torchvision.datasets.CIFAR10
    dataset = torchvision.datasets.ImageNet(root=args.data_path, split='val')
    subset_indices = np.random.choice(len(dataset), args.batch_size, replace=False)
    test_subset = torch.utils.data.Subset(dataset, subset_indices)
    
    # 简单的内部 Dataset 包装器应用 transform
    class TestDataset(torch.utils.data.Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
        def __getitem__(self, idx):
            x, y = self.subset[idx]
            return self.transform(x), y
        def __len__(self):
            return len(self.subset)

    test_dataset = TestDataset(test_subset, transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Registering hooks for layer {args.layer_id}...")
    value_hooks, value_outputs = insert_v_hooks_for_test(model, args.layer_id, P)

    print("Running forward pass...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(args.device)
            model(images)
            break  # 只需运行一个 batch 获取统计信息

    # 移除 hooks
    for hook in value_hooks:
        hook.remove()

    l2_results = value_outputs[0]
    
    # Min-Max 归一化
    norm_l2_results = (l2_results - l2_results.min()) / (l2_results.max() - l2_results.min())

    print("Plotting results...")
    plot_title = f"PCA Component Norms on Layer {args.layer_id} (Value)"
    plot_pca_norms_subplots(norm_l2_results, title_prefix=plot_title, save_path=args.save_plot_path)

if __name__ == "__main__":
    main()