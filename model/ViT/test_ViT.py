#!/usr/bin/env python
# coding: utf-8

import os
import sys
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ViTForImageClassification
from transkv import transkv_vit
from utils import compute_pruned_dim

def get_imagenet_val_loader(data_path, batch_size, num_workers, seed=42):
    """
    构建 ImageNet 验证集 DataLoader (基于原 ImageBuilder 简化)
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = torchvision.datasets.ImageNet(root=data_path, split='val', transform=transform)
    
    # 为了保证可复现性设置 generator
    generator = torch.Generator().manual_seed(seed)
    
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True,
        generator=generator
    )
    return loader

def evaluate_vit(model, test_loader, device, criterion=None):
    """
    评估模型在给定验证集上的性能 (基于原 vit_evaluation 简化)
    """
    model.eval()
    
    global_correct = 0
    global_total = 0
    eval_loss = 0.0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)

            model_outputs = model(images)
            
            # 兼容直接输出或者含有 logits 属性的输出
            if hasattr(model_outputs, 'logits'):
                logits = model_outputs.logits
            else:
                logits = model_outputs

            if criterion is not None:
                loss = criterion(logits, labels)
                eval_loss += loss.item()

            _, preds = logits.max(1)

            correct = (preds == labels)
            global_correct += correct.sum().item()
            global_total += labels.size(0)

    result = {}
    
    if criterion is not None:
        eval_loss /= len(test_loader)
        result['loss'] = eval_loss
        
    total_acc = global_correct / global_total
    result['total'] = total_acc

    return result

def main():
    parser = argparse.ArgumentParser(description="Test ViT model on ImageNet (Optional: Pruning)")
    
    # 基础模型与设备配置
    parser.add_argument("--model_name", type=str, default="google/vit-base-patch16-224", help="HuggingFace model name")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    
    # 数据集配置
    parser.add_argument("--data_path", type=str, required=True, help="Root directory for ImageNet dataset")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for validation")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for data loading")
    
    # 剪枝相关参数
    parser.add_argument("--pca_path", type=str, default=None, help="Path to the PCA .pt file. If None, test original model.")
    parser.add_argument("--pruned_ratio", type=float, default=0.5, help="Ratio of pruned dimensions if pruning")

    args = parser.parse_args()

    # 加载模型
    print(f"Loading original model '{args.model_name}' to {args.device}...")
    model = ViTForImageClassification.from_pretrained(args.model_name)
    model.to(args.device)

    args.pruned_dim = compute_pruned_dim(model.vit, args.pruned_ratio)

    # 判断是否需要剪枝
    if args.pca_path is not None and args.pca_path.strip() != "":
        print(f"PCA save path provided: {args.pca_path}")
        print(f"Pruning model to head dimension: {args.pruned_dim}...")
        # 调用分离出来的剪枝逻辑，由于 transkv_vit 接受 original_model 和 args，直接传入
        model = transkv_vit(model, args)
        model.to(args.device)
        print("Model pruning completed.")
    else:
        print("No PCA save path provided. Testing the original unpruned model.")

    # 加载数据集
    print(f"Initializing ImageNet validation loader from {args.data_path}...")
    test_loader = get_imagenet_val_loader(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # 评估模型
    criterion = torch.nn.CrossEntropyLoss()
    print("Starting evaluation...")
    results = evaluate_vit(model, test_loader, args.device, criterion)

    # 输出结果
    print("\n" + "="*30)
    print("Evaluation Results:")
    print(f"Best Accuracy: {results['total']*100:.2f}%")
    print("="*30 + "\n")

if __name__ == "__main__":
    main()