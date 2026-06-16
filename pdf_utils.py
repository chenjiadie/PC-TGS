import torch
from torch.distributions import MultivariateNormal

def multivariate_normal_pdf_origin(mu, cov):
    """
    批量计算二维高斯分布在 x = [0,0] 的 PDF (支持 N,B 维度 + CUDA + 反向传播)
    mu  : (N, B, 2)     均值
    cov : (N, B, 2, 2)  协方差矩阵
    return: (N, B)      对应每个分布的 PDF 值
    """
    # 协方差矩阵求逆和行列式
    inv_cov = torch.linalg.inv(cov)                       # (N, B, 2, 2)
    det_cov = torch.linalg.det(cov)                            # (N, B)

    # 常数项 1 / (2π√|Σ|)
    norm_const = 1.0 / (2 * torch.pi * torch.sqrt(det_cov))  # (N, B)

    # (x - μ) = -μ
    diff = (-mu).unsqueeze(-2)                          # (N, B, 1, 2)

    # 二次型 (x-μ)^T Σ^{-1} (x-μ)
    quad_form = torch.matmul(torch.matmul(diff, inv_cov), diff.transpose(-1, -2)).squeeze(-1).squeeze(-1)  # (N, B)

    # exp(-0.5 * quad_form)
    exp_term = torch.exp(-0.5 * quad_form)

    return norm_const * exp_term

# import torch

# def multivariate_normal_pdf(mu, cov, jitter=1e-6):
#     """
#     批量计算二维高斯分布在 x = [0,0] 的 PDF (数值稳定版本)
#     mu    : (N, B, 2)     均值
#     cov   : (N, B, 2, 2)  协方差矩阵
#     jitter: 正则化系数，用于稳定协方差矩阵
#     return: (N, B)        PDF 值
#     """
#     # 添加正则化确保正定性
#     I = torch.eye(2, device=cov.device, dtype=cov.dtype).view(1, 1, 2, 2)
#     cov_reg = cov + jitter * I
    
#     # Cholesky分解 (L: 下三角矩阵)
#     # try:
#     L = torch.linalg.cholesky(cov_reg)  # (N, B, 2, 2)
#     # except RuntimeError as e:
#         # 添加更详细错误信息
#         # raise RuntimeError(f"Cholesky分解失败，最小特征值: {torch.linalg.eigvalsh(cov_reg).min().item()}") from e
    
#     # 计算行列式 (利用Cholesky因子)
#     diag_L = torch.diagonal(L, dim1=-2, dim2=-1)  # (N, B, 2)
#     det_cov = torch.prod(diag_L, dim=-1) ** 2     # (N, B)
    
#     # 常数项 1/(2π√|Σ|)
#     norm_const = 1.0 / (2 * torch.pi * torch.sqrt(det_cov))  # (N, B)
    
#     # 计算二次型: (x-μ)^T Σ^{-1} (x-μ) = ||L^{-1}(x-μ)||^2
#     diff = -mu  # (N, B, 2)
#     y = torch.linalg.solve_triangular(L, diff.unsqueeze(-1), upper=False)  # (N, B, 2, 1)
#     quad_form = torch.sum(y.squeeze(-1)**2, dim=-1)  # (N, B)
    
#     # 计算指数项 (数值稳定)
#     exp_term = torch.exp(-0.5 * quad_form)
    
#     return norm_const * exp_term


def multivariate_normal_pdf_stable(mu, cov, jitter=1e-6):
    """
    使用 PyTorch 内置函数计算二维高斯分布在 x=[0,0] 的 PDF (数值稳定版本)
    mu    : (N, B, 2)     均值
    cov   : (N, B, 2, 2)  协方差矩阵
    jitter: 正则化系数，用于稳定协方差矩阵 (默认 1e-6)
    return: (N, B)        PDF 值
    """
    # 保存原始形状
    original_shape = mu.shape[:-1]
    d = mu.shape[-1]
    
    # 重塑为批量形式 (N*B, 2)
    mu_flat = mu.reshape(-1, d)
    
    # 重塑协方差矩阵并添加正则化
    cov_flat = cov.reshape(-1, d, d)
    I = torch.eye(d, device=cov.device, dtype=cov.dtype).unsqueeze(0)
    cov_reg = cov_flat + jitter * I
    
    # 确保协方差矩阵对称
    cov_sym = 0.5 * (cov_reg + cov_reg.transpose(-1, -2))
    
    # 创建多元正态分布
    try:
        dist = MultivariateNormal(loc=mu_flat, covariance_matrix=cov_sym)
    except RuntimeError as e:
        # 添加诊断信息
        min_eigvals = torch.linalg.eigvalsh(cov_sym).min(dim=-1)[0]
        raise RuntimeError(
            f"协方差矩阵非正定! 最小特征值: {min_eigvals.min().item()}, "
            f"位置: {min_eigvals.argmin().item()}"
        ) from e
    
    # 在 x=[0,0] 处计算 PDF
    x = torch.zeros_like(mu_flat)
    pdf_flat = torch.exp(dist.log_prob(x))
    
    # 恢复原始形状
    return pdf_flat.reshape(original_shape)


if __name__ == "__main__":
# ===== 测试 GPU + 反向传播 =====
    device = "cuda" if torch.cuda.is_available() else "cpu"

    N, B = 4, 5
    mu  = torch.randn(N, B, 2, device=device, requires_grad=True)
    cov = torch.eye(2, device=device).expand(N, B, 2, 2).clone().requires_grad_()

    pdf_vals = multivariate_normal_pdf_origin(mu, cov)  # (N,B)
    loss = pdf_vals.sum()
    loss.backward()

    print("PDF values:", pdf_vals)
    print("Grad mu shape:", mu.grad.shape)
    print("Grad cov shape:", cov.grad.shape)
