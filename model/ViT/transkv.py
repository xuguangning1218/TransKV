import torch
import copy
from tqdm import tqdm

def get_pca_matrices(pca_save_path, device, pruned_dim):
    """
    读取 PCA 结果并直接提取 K 和 V 的投影矩阵 (截断到 pruned_dim)
    """
    pca_qkvo_outputs = torch.load(pca_save_path, map_location=device)

    # 仅提取 K 和 V
    pca_key_outputs = pca_qkvo_outputs['key']
    pca_value_outputs = pca_qkvo_outputs['value']

    P_K_layers = []
    P_V_layers = []

    for layer_id in range(len(pca_key_outputs)):
        # 处理 K 的 PCA 矩阵
        if isinstance(pca_key_outputs[layer_id], list) and len(pca_key_outputs[layer_id]) > 0:
            P_K = torch.stack(pca_key_outputs[layer_id])[:, :, :pruned_dim]
        else:
            P_K = pca_key_outputs[layer_id][:, :, :pruned_dim]
            
        # 处理 V 的 PCA 矩阵
        if isinstance(pca_value_outputs[layer_id], list) and len(pca_value_outputs[layer_id]) > 0:
            P_V = torch.stack(pca_value_outputs[layer_id])[:, :, :pruned_dim]
        else:
            P_V = pca_value_outputs[layer_id][:, :, :pruned_dim]

        P_K_layers.append(P_K.float())
        P_V_layers.append(P_V.float())
    
    # 转换为 Tensor: [num_layers, num_heads, head_dim, pruned_dim]
    P_K_layers = torch.stack(P_K_layers, dim=0)
    P_V_layers = torch.stack(P_V_layers, dim=0)

    return P_K_layers, P_V_layers


def transkv_vit(original_model, args):
    model = copy.deepcopy(original_model)

    config = model.config
    num_attention_heads = config.num_attention_heads      # 注意力头数
    head_dim = config.hidden_size // num_attention_heads  # 每个头的维度
    pruned_dim = min(args.pruned_dim, head_dim)           # 确保剪枝维度不超过头维度
    hidden_size = config.hidden_size                      # ViT 的隐藏维度

    # 获取全局的 K 和 V 投影矩阵
    P_K_layers, P_V_layers = get_pca_matrices(args.pca_path, args.device, pruned_dim)
    
    # 遍历每一层进行剪枝
    for i in tqdm(range(config.num_hidden_layers), desc="Pruning layers"):
        self_attn = model.vit.encoder.layer[i].attention

        # 获取当前层的 K 和 V 投影矩阵: [num_heads, head_dim, pruned_dim]
        P_K = P_K_layers[i].to(args.device)
        P_V = P_V_layers[i].to(args.device)

        # 提取原始权重和偏置
        q_weight = self_attn.attention.query.weight.data.reshape(num_attention_heads, head_dim, hidden_size)
        q_bias = self_attn.attention.query.bias.data.reshape(num_attention_heads, head_dim, 1)
        k_weight = self_attn.attention.key.weight.data.reshape(num_attention_heads, head_dim, hidden_size)
        k_bias = self_attn.attention.key.bias.data.reshape(num_attention_heads, head_dim, 1)
        v_weight = self_attn.attention.value.weight.data.reshape(num_attention_heads, head_dim, hidden_size)
        v_bias = self_attn.attention.value.bias.data.reshape(num_attention_heads, head_dim, 1)
        o_weight = self_attn.output.dense.weight.data.reshape(hidden_size, num_attention_heads, head_dim)
        
        # 预计算转置矩阵: [num_heads, pruned_dim, head_dim]
        P_K_T = P_K.transpose(-1, -2)
        P_V_T = P_V.transpose(-1, -2)

        # 使用 P_K 投影 Query 和 Key (左乘 P_K^T)
        pca_q_weight = torch.bmm(P_K_T, q_weight)
        pca_q_bias = torch.bmm(P_K_T, q_bias)
        pca_k_weight = torch.bmm(P_K_T, k_weight)
        pca_k_bias = torch.bmm(P_K_T, k_bias)

        # 使用 P_V 投影 Value (左乘 P_V^T)
        pca_v_weight = torch.bmm(P_V_T, v_weight)
        pca_v_bias = torch.bmm(P_V_T, v_bias)

        # 使用 P_V 投影 Output (右乘 P_V)
        # o_weight: [hidden_size, num_heads, head_dim]
        # P_V: [num_heads, head_dim, pruned_dim]
        # 输出: [hidden_size, num_heads, pruned_dim]
        pca_o_weight = torch.einsum("nhd,hdp->nhp", o_weight, P_V)

        # 将更新后的权重重新赋值给模型
        self_attn.attention.query.weight.data = pca_q_weight.reshape(num_attention_heads * pruned_dim, hidden_size).contiguous()
        self_attn.attention.query.bias.data = pca_q_bias.reshape(num_attention_heads * pruned_dim).contiguous()
        self_attn.attention.key.weight.data = pca_k_weight.reshape(num_attention_heads * pruned_dim, hidden_size).contiguous()
        self_attn.attention.key.bias.data = pca_k_bias.reshape(num_attention_heads * pruned_dim).contiguous()
        self_attn.attention.value.weight.data = pca_v_weight.reshape(num_attention_heads * pruned_dim, hidden_size).contiguous()
        self_attn.attention.value.bias.data = pca_v_bias.reshape(num_attention_heads * pruned_dim).contiguous()
        self_attn.output.dense.weight.data = pca_o_weight.reshape(hidden_size, num_attention_heads * pruned_dim).contiguous()

        # 更新模型参数配置
        self_attn.attention.attention_head_size = pruned_dim
        self_attn.attention.all_head_size = num_attention_heads * pruned_dim

    return model