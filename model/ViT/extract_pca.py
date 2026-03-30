#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import torch
import torchvision
import numpy as np
from tqdm import tqdm
from transformers import ViTForImageClassification
from utils import pca_calc, set_seed

class TransformedSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

def get_dataloader(data_path, sample_ratio, batch_size, num_workers):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = torchvision.datasets.ImageNet(root=data_path, split='train')
    train_indices = list(range(len(dataset)))
    targets = dataset.targets

    if sample_ratio < 1.0:
        num_classes = 1000
        class_to_indices = {i: [] for i in range(num_classes)}
        for idx, label in zip(train_indices, targets):
            class_to_indices[label].append(idx)

        sampled_indices = []
        for class_id in class_to_indices:
            class_indices = class_to_indices[class_id]
            sample_size = max(1, int(len(class_indices) * sample_ratio))
            sampled_class_indices = np.random.choice(class_indices, size=sample_size, replace=False)
            sampled_indices.extend(sampled_class_indices)
        print(f"Original training samples: {len(train_indices)}")
        print(f"Sampled training samples: {len(sampled_indices)}")
    else:
        sampled_indices = train_indices
        print(f"Original training samples: {len(train_indices)}")

    train_subset = torch.utils.data.Subset(dataset, sampled_indices)
    train_dataset = TransformedSubset(train_subset, transform=transform)
    loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    return loader

# ================= 定义钩子 =================
def insert_hooks(model, layer_id, target_types):
    hooks = []
    outputs = {t: {} for t in target_types}
    self_attn = model.vit.encoder.layer[layer_id].attention

    def make_hook_fn(target_type):
        def hook_fn(module, input, output, index, num_attention_heads, attention_head_size):
            if index not in outputs[target_type]:
                outputs[target_type][index] = []
            batch_size, seq_num, _ = output.shape
            outputs[target_type][index].append(output.reshape(batch_size, seq_num, num_attention_heads, attention_head_size).to('cpu'))
        return hook_fn

    num_attention_heads = self_attn.attention.num_attention_heads
    attention_head_size = self_attn.attention.attention_head_size

    if "query" in target_types and hasattr(self_attn.attention, "query"):
        hook = self_attn.attention.query.register_forward_hook(
            lambda m, i, o, idx=layer_id, n=num_attention_heads, s=attention_head_size: make_hook_fn("query")(m, i, o, idx, n, s))
        hooks.append(hook)
    if "key" in target_types and hasattr(self_attn.attention, "key"):
        hook = self_attn.attention.key.register_forward_hook(
            lambda m, i, o, idx=layer_id, n=num_attention_heads, s=attention_head_size: make_hook_fn("key")(m, i, o, idx, n, s))
        hooks.append(hook)
    if "value" in target_types and hasattr(self_attn.attention, "value"):
        hook = self_attn.attention.value.register_forward_hook(
            lambda m, i, o, idx=layer_id, n=num_attention_heads, s=attention_head_size: make_hook_fn("value")(m, i, o, idx, n, s))
        hooks.append(hook)
    if "output" in target_types and hasattr(self_attn, "output"):
        hook = self_attn.output.dense.register_forward_hook(
            lambda m, i, o, idx=layer_id, n=num_attention_heads, s=attention_head_size: make_hook_fn("output")(m, i, o, idx, n, s))
        hooks.append(hook)

    return hooks, outputs

# ================= 提取Attention输出 =================
def process_targets(model, dataloader, device, target_types):
    pca_results = {t: [] for t in target_types}
    
    for layer_id in range(len(model.vit.encoder.layer)):
        print(f"Processing layer {layer_id} for {target_types}...")
        hooks, outputs = insert_hooks(model, layer_id, target_types)
        num_attention_heads = model.vit.encoder.layer[layer_id].attention.attention.num_attention_heads

        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc=f"Layer {layer_id} Forward"):
                images = images.to(device)
                model(images)
                torch.cuda.empty_cache()

        for hook in hooks:
            hook.remove()

        for t in target_types:
            pca_res = pca_calc(outputs[t][layer_id], device, num_attention_heads)
            pca_results[t].append(pca_res)
            del outputs[t][layer_id]
        
        torch.cuda.empty_cache()

    return pca_results

def main():
    parser = argparse.ArgumentParser(description="ImageNet PCA Constructor for ViT")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (e.g., cuda:1)")
    parser.add_argument("--model_name", type=str, default="google/vit-base-patch16-224", help="HuggingFace model name")
    parser.add_argument("--data_path", type=str, required=True, help="Path to ImageNet dataset")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save PCA pt files")
    parser.add_argument("--sample_ratio", type=float, default=0.05, help="Ratio of dataset to sample")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for dataloader")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for dataloader")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    if args.sample_ratio < 1:
        pca_save_path = os.path.join(args.save_dir, f'ImageNet_{int(args.sample_ratio*100)}.pt')
    else:
        pca_save_path = os.path.join(args.save_dir, 'ImageNet.pt')

    print(f"Loading model {args.model_name} to {args.device}...")
    model = ViTForImageClassification.from_pretrained(args.model_name)
    model.to(args.device)
    model.eval()

    print("Initializing DataLoader...")
    dataloader = get_dataloader(args.data_path, args.sample_ratio, args.batch_size, args.num_workers)

    # 处理 Key 和 Value
    print("=== Starting Key-Value Phase ===")
    kv_results = process_targets(model, dataloader, args.device, ["key", "value"])
    
    # 保存
    print("=== Combining and Saving Results ===")
    qkvo_combined = {
        "key": kv_results["key"],
        "value": kv_results["value"],
    }
    
    torch.save(qkvo_combined, pca_save_path)
    print(f"Successfully saved PCA rotation matrices to {pca_save_path}")

if __name__ == "__main__":
    main()