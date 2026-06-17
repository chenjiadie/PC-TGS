import torch
import pdb

def filter_by_mahalanobis(mu_2d, cov_2d,sorted_weight, tau=3.0, eps=1e-6):
    """
    并行版本 + 数值稳定处理
    """
    N, M, _ = mu_2d.shape

    # 对称化并加正则，防止协方差奇异
    cov_2d_sym = 0.5 * (cov_2d + cov_2d.transpose(-1, -2))
    cov_2d_reg = cov_2d_sym + eps * torch.eye(2, device=cov_2d.device).expand(N, M, 2, 2)

    cov2d_inv = torch.linalg.inv(cov_2d_reg)  # (N, M, 2, 2)

    diff_col = mu_2d.unsqueeze(-1)  # (N, M, 2, 1)
    left_term = torch.matmul(cov2d_inv, diff_col)
    D2 = torch.matmul(diff_col.transpose(-1, -2), left_term).squeeze(-1).squeeze(-1)  # (N, M)

    # 限制数值，避免梯度爆炸
    D2 = torch.clamp(D2, min=0.0, max=1e6)

    mask = (D2 < tau**2).float()  # (N, M)

    # 获取排序索引
    sort_idx = torch.argsort(mask, dim=1, descending=True)  # (N, M)

    # 批量 gather
    batch_idx = torch.arange(N, device=mu_2d.device).unsqueeze(-1).expand(N, M)
    mu_sorted = mu_2d[batch_idx, sort_idx]        # (N, M, 2)
    cov_sorted = cov_2d[batch_idx, sort_idx]      # (N, M, 2, 2)
    mask_sorted = mask[batch_idx, sort_idx]  # (N, M, 1)
    weight_sorted = sorted_weight[batch_idx, sort_idx]  # (N, M)

    # 末尾补零
    mu_2d_new = mu_sorted * mask_sorted.unsqueeze(-1)
    cov_2d_new = cov_sorted * mask_sorted.unsqueeze(-1).unsqueeze(-1)
    weight_new = weight_sorted * mask_sorted  # (N, M)

    return weight_new


# 测试
if __name__ == "__main__":
    N, M = 2, 5
    mu_2d = torch.randn(N, M, 2, requires_grad=True)
    cov_2d = torch.eye(2).expand(N, M, 2, 2).clone().requires_grad_()
    sorted_weight = torch.rand(N, M, requires_grad=True)
    tau = 1.5

    mu_new, cov_new,weight_new = filter_by_mahalanobis(mu_2d, cov_2d, sorted_weight,tau)
    for i in range(N):
        print("mu_2d_new:", mu_new[i])  # (N, M, 2)
        print("cov_2d_new:", cov_new[i])
        print("sorted_weight:", weight_new[i])

    # 反向传播测试
    loss = mu_new.sum() + cov_new.sum()+weight_new.sum()
    loss.backward()
    print("mu_2d.grad shape:", mu_2d.grad.shape)
    print("cov_2d.grad shape:", cov_2d.grad.shape)
    print("sorted_weight.grad shape:", sorted_weight.grad.shape)
