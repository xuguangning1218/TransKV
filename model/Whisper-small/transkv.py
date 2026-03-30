import torch
import copy
from tqdm import tqdm

def get_pca_matrices_whisper(pca_path, device, pruned_dim):
    """
    读取 Whisper 的 PCA 结果，并直接提取 K 和 V 的投影矩阵 (截断到 pruned_dim)。
    返回格式:
        matrices = {
            'encoder_self': (P_K_layers, P_V_layers),
            'decoder_self': (P_K_layers, P_V_layers),
            'decoder_cross': (P_K_layers, P_V_layers)
        }
    """
    pca_outputs = torch.load(pca_path, map_location=device)
    print("pca_outputs keys:", pca_outputs.keys())
    
    def extract_component_matrices(component_name):
        P_K_layers = []
        P_V_layers = []
        
        # 兼容新的 extract_pca_whisper.py 格式和旧格式
        if component_name in pca_outputs:
            keys = pca_outputs[component_name]['key']
            values = pca_outputs[component_name]['value']
            
        for layer_id in range(len(keys)):
            P_K = keys[layer_id]
            P_V = values[layer_id]
            
            if isinstance(P_K, list) and len(P_K) > 0:
                P_K = torch.stack(P_K)
            if isinstance(P_V, list) and len(P_V) > 0:
                P_V = torch.stack(P_V)
                
            P_K_layers.append(P_K[:, :, :pruned_dim].float())
            P_V_layers.append(P_V[:, :, :pruned_dim].float())
            
        # 转换为 Tensor: [num_layers, num_heads, head_dim, pruned_dim]
        return torch.stack(P_K_layers, dim=0), torch.stack(P_V_layers, dim=0)

    matrices = {
        'encoder': extract_component_matrices('encoder'),
        'decoder': extract_component_matrices('decoder'),
        'cross': extract_component_matrices('cross')
    }
    
    return matrices

def pruning_action_whisper(model_layers, P_K_layers, P_V_layers, num_heads, hidden_size, pruned_dim, device, is_cross_attn=False, desc="Pruning"):
    """
    针对指定的 Transformer 层列表（Encoder 或 Decoder）执行张量化的 K/V 投影。
    """
    for layer_id in tqdm(range(len(model_layers)), desc=desc):
        # Whisper 的 decoder 层既有 self_attn 又有 encoder_attn (即 cross attention)
        if is_cross_attn:
            attn_module = model_layers[layer_id].encoder_attn
        else:
            attn_module = model_layers[layer_id].self_attn

        head_dim = hidden_size // num_heads
        
        # 获取当前层的 P_K 和 P_V 矩阵: [num_heads, head_dim, pruned_dim]
        P_K = P_K_layers[layer_id].to(device)
        P_V = P_V_layers[layer_id].to(device)
        
        # 预计算转置矩阵: [num_heads, pruned_dim, head_dim]
        P_K_T = P_K.transpose(-1, -2)
        P_V_T = P_V.transpose(-1, -2)

        # ---------------- 投影 Query ----------------
        # q_weight 原始 shape: [embed_dim, hidden_size] -> [num_heads, head_dim, hidden_size]
        q_weight = attn_module.q_proj.weight.data.reshape(num_heads, head_dim, hidden_size)
        pca_q_weight = torch.bmm(P_K_T, q_weight)
        attn_module.q_proj.weight.data = pca_q_weight.reshape(num_heads * pruned_dim, hidden_size).contiguous()
        
        if hasattr(attn_module.q_proj, 'bias') and attn_module.q_proj.bias is not None:
            q_bias = attn_module.q_proj.bias.data.reshape(num_heads, head_dim, 1)
            pca_q_bias = torch.bmm(P_K_T, q_bias)
            attn_module.q_proj.bias.data = pca_q_bias.reshape(num_heads * pruned_dim).contiguous()

        # ---------------- 投影 Key ----------------
        k_weight = attn_module.k_proj.weight.data.reshape(num_heads, head_dim, hidden_size)
        pca_k_weight = torch.bmm(P_K_T, k_weight)
        attn_module.k_proj.weight.data = pca_k_weight.reshape(num_heads * pruned_dim, hidden_size).contiguous()
        
        if hasattr(attn_module.k_proj, 'bias') and attn_module.k_proj.bias is not None:
            k_bias = attn_module.k_proj.bias.data.reshape(num_heads, head_dim, 1)
            pca_k_bias = torch.bmm(P_K_T, k_bias)
            attn_module.k_proj.bias.data = pca_k_bias.reshape(num_heads * pruned_dim).contiguous()

        # ---------------- 投影 Value ----------------
        v_weight = attn_module.v_proj.weight.data.reshape(num_heads, head_dim, hidden_size)
        pca_v_weight = torch.bmm(P_V_T, v_weight)
        attn_module.v_proj.weight.data = pca_v_weight.reshape(num_heads * pruned_dim, hidden_size).contiguous()
        
        if hasattr(attn_module.v_proj, 'bias') and attn_module.v_proj.bias is not None:
            v_bias = attn_module.v_proj.bias.data.reshape(num_heads, head_dim, 1)
            pca_v_bias = torch.bmm(P_V_T, v_bias)
            attn_module.v_proj.bias.data = pca_v_bias.reshape(num_heads * pruned_dim).contiguous()

        # ---------------- 投影 Output ----------------
        # o_weight 原始 shape: [hidden_size, embed_dim] -> [hidden_size, num_heads, head_dim]
        out_weight = attn_module.out_proj.weight.data.reshape(hidden_size, num_heads, head_dim)
        # einsum 实现右乘 P_V: [hidden_size, num_heads, head_dim] @ [num_heads, head_dim, pruned_dim] 
        pca_out_weight = torch.einsum("nhd,hdp->nhp", out_weight, P_V)
        attn_module.out_proj.weight.data = pca_out_weight.reshape(hidden_size, num_heads * pruned_dim).contiguous()

        # ---------------- 更新模型结构属性 ----------------
        attn_module.head_dim = pruned_dim
        attn_module.embed_dim = num_heads * pruned_dim
        attn_module.q_proj.out_features = attn_module.embed_dim
        attn_module.k_proj.out_features = attn_module.embed_dim
        attn_module.v_proj.out_features = attn_module.embed_dim
        attn_module.out_proj.in_features = attn_module.embed_dim

def cloverpca_whisper(original_model, args):
    """
    Whisper PCA 剪枝主函数
    """
    model = copy.deepcopy(original_model)
    config = model.config
    
    hidden_size = config.d_model
    num_encoder_heads = config.encoder_attention_heads
    num_decoder_heads = config.decoder_attention_heads
    
    # 假设 encoder 和 decoder 的原始 head_dim 相同
    head_dim = hidden_size // num_encoder_heads
    pruned_dim = min(args.pruned_dim, head_dim)

    # 1. 统一加载 PCA 矩阵
    pca_matrices = get_pca_matrices_whisper(args.pca_path, args.device, pruned_dim)

    # 2. 根据 args 指定的组件进行相应的剪枝
    component_config = getattr(args, 'whisper_pruning_component', 'cross')

    if component_config in ['encoder', 'cross']:
        P_K_enc, P_V_enc = pca_matrices['encoder']
        pruning_action_whisper(
            model_layers=model.model.encoder.layers,
            P_K_layers=P_K_enc, P_V_layers=P_V_enc,
            num_heads=num_encoder_heads, hidden_size=hidden_size, 
            pruned_dim=pruned_dim, device=args.device,
            is_cross_attn=False, desc="Pruning Encoder Self-Attn"
        )

    if component_config in ['decoder', 'cross']:
        P_K_dec_self, P_V_dec_self = pca_matrices['decoder']
        pruning_action_whisper(
            model_layers=model.model.decoder.layers,
            P_K_layers=P_K_dec_self, P_V_layers=P_V_dec_self,
            num_heads=num_decoder_heads, hidden_size=hidden_size, 
            pruned_dim=pruned_dim, device=args.device,
            is_cross_attn=False, desc="Pruning Decoder Self-Attn"
        )
        
        P_K_dec_cross, P_V_dec_cross = pca_matrices['cross']
        pruning_action_whisper(
            model_layers=model.model.decoder.layers,
            P_K_layers=P_K_dec_cross, P_V_layers=P_V_dec_cross,
            num_heads=num_decoder_heads, hidden_size=hidden_size, 
            pruned_dim=pruned_dim, device=args.device,
            is_cross_attn=True, desc="Pruning Decoder Cross-Attn"
        )

    return model