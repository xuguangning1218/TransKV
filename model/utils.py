import math
import random
import torch
import numpy as np

def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@torch.no_grad()
def pca_calc(X: list[torch.Tensor], device: str, num_heads: int) -> torch.Tensor:
    P_list = []
    for h in range(num_heads):
        H = None
        for idx, X_batch in enumerate(X):

            X_batch = X_batch[:, :, h, :].double().to(device)
            H_batch = torch.sum(X_batch.mT @ X_batch, dim=0)  # sum over the batch dimension.
            H = H_batch if H is None else H + H_batch

        damp = 0.01 * torch.mean(torch.diag(H))
        diag = torch.arange(H.shape[-1]).to(device)
        H[diag, diag] = H[diag, diag] + damp
        X_eig = torch.linalg.eigh(H)
        del H
        index = torch.argsort(X_eig[0], descending=True)
        eigen_vec = X_eig[1][:, index]
        P_list.append(eigen_vec.float().to(device))
    return P_list

def compute_pruned_dim(model, pruned_ratio=0.1):
    config = model.config
    head_dim = config.hidden_size // config.num_attention_heads
    print(f"Original head dimension: {head_dim}, Pruned ratio: {pruned_ratio}")
    if pruned_ratio == 0.0:
        return head_dim
    else:
        return head_dim - math.ceil(head_dim * pruned_ratio)