import torch
from tqdm import tqdm

def transkv_llama3(original_model, args):
    """
    针对 Llama3-8B 模型的特征剪枝逻辑 (仅剪枝 Value 和 Output，跳过 Query 和 Key（因为RoPE需要在线剪枝）)
    """
    # 如果不需要保留原模型，直接在原模型上修改即可节省显存
    model = original_model
    config = model.config
    
    hidden_size = config.hidden_size
    num_attention_heads = config.num_attention_heads      # 注意力头数 (Llama3-8B 为 32)
    num_key_value_heads = config.num_key_value_heads      # KV 头数 (Llama3-8B 为 8)
    group_size = num_attention_heads // num_key_value_heads # GQA 分组大小 (4)
    head_dim = config.hidden_size // config.num_attention_heads # 每个头的维度 (128)
    
    pruned_dim = min(args.pruned_dim, head_dim) # 确保剪枝维度不超过原头维度
    
    # 加载 PCA 矩阵
    pca_qkvo_outputs = torch.load(args.pca_path, map_location=args.device)
    pca_value_outputs = pca_qkvo_outputs['value']
    
    # 提前处理 PCA 矩阵的截断
    for layer_id in range(len(pca_value_outputs)):
        if isinstance(pca_value_outputs[layer_id], list):
            pca_value_outputs[layer_id] = torch.stack(pca_value_outputs[layer_id])[:, :, :pruned_dim]
        else:
            pca_value_outputs[layer_id] = pca_value_outputs[layer_id][:, :, :pruned_dim]
    
    # 遍历每一层进行剪枝
    for layer_id in tqdm(range(config.num_hidden_layers), desc="Pruning layers"):
        self_attn = model.model.layers[layer_id].self_attn
        
        # 提取原始权重
        v_weight = self_attn.v_proj.weight.data.reshape(num_key_value_heads, head_dim, hidden_size).to(args.device)
        o_weight = self_attn.o_proj.weight.data.reshape(hidden_size, num_key_value_heads, group_size, head_dim).to(args.device)

        # 获取当前层的 PCA Value 投影矩阵: [num_key_value_heads, head_dim, pruned_dim]
        pca_value_output = pca_value_outputs[layer_id].to(dtype=v_weight.dtype, device=args.device)

        # ================= 投影 Value =================
        pca_value_weight = []
        for g in range(num_key_value_heads):            
            W_V_g = v_weight[g].transpose(0, 1) # (hidden_size, head_dim)
            pca_v_g = torch.matmul(W_V_g, pca_value_output[g]) # (hidden_size, head_dim) @ (head_dim, pruned_dim) -> (hidden_size, pruned_dim)
            pca_value_weight.append(pca_v_g.transpose(0, 1))  # (pruned_dim, hidden_size)
            
        pca_value_weight = torch.stack(pca_value_weight, dim=0)
        self_attn.v_proj.weight.data = pca_value_weight.reshape(num_key_value_heads * pruned_dim, hidden_size).contiguous()

        # ================= 投影 Output =================
        pca_output_weight = []
        for g in range(num_key_value_heads):
            W_O_g = o_weight[:, g]  # (hidden_size, group_size, head_dim)
            pca_O_g = torch.matmul(W_O_g, pca_value_output[g])  # (hidden_size, group_size, head_dim) @ (head_dim, pruned_dim) -> (hidden_size, group_size, pruned_dim)
            pca_output_weight.append(pca_O_g)  
            
        pca_output_weight = torch.stack(pca_output_weight, dim=1)  # (hidden_size, num_key_value_heads, group_size, pruned_dim)
        self_attn.o_proj.weight.data = pca_output_weight.reshape(hidden_size, num_key_value_heads * group_size * pruned_dim).contiguous()

        # ================= 更新模型配置属性 =================
        self_attn.v_proj.out_features = self_attn.v_proj.weight.shape[0] 
        self_attn.o_proj.in_features = self_attn.o_proj.weight.shape[1]
        self_attn.head_dim = pruned_dim
    
    return model