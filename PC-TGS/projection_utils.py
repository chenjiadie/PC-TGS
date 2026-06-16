import numpy as np
import torch
import pdb


import torch

def sph_to_cart(theta, phi):
    """
    球坐标转笛卡尔坐标 (支持批处理)
    输入:
        theta: [batch_size, ...] 极角 (弧度)
        phi:   [batch_size, ...] 方位角 (弧度)
    返回:
        cart: [batch_size, ..., 3] 笛卡尔坐标 (x,y,z)
    """

    sin_theta = torch.sin(theta)
    x = sin_theta * torch.cos(phi)
    y = sin_theta * torch.sin(phi)
    z = torch.cos(theta)
    

    return torch.stack((x, y, z), dim=-1)

def tangent_basis(P_BS,theta0, phi0):
    """
    计算球面上点的切空间基 (支持批处理)
    输入:
        theta0: [batch_size] 极角 (弧度)
        phi0:   [batch_size] 方位角 (弧度)
    返回:
        n0: [batch_size, 3] 法向量
        u1: [batch_size, 3] 第一切向量
        u2: [batch_size, 3] 第二切向量
        U:  [batch_size, 3, 2] 切空间基矩阵
    """

    n0 = sph_to_cart(theta0, phi0)+P_BS[None,:]  # [batch_size, 3]
    

    cos_theta = torch.cos(theta0)
    sin_theta = torch.sin(theta0)
    cos_phi = torch.cos(phi0)
    sin_phi = torch.sin(phi0)
    

    u1 = torch.stack([
        cos_theta * cos_phi,
        cos_theta * sin_phi,
        -sin_theta
    ], dim=-1)  # [batch_size, 3]
    
    u2 = torch.stack([
        -sin_phi,
        cos_phi,
        torch.zeros_like(theta0)  
    ], dim=-1)  # [batch_size, 3]
    
    #  [batch_size, 3, 2]
    U = torch.stack([u1, u2], dim=-1)
    
    norm = torch.linalg.norm(U, dim=1, keepdim=True)
    U = U / (norm + 1e-8)  # 添加小常数防止除零
    
    return n0, u1, u2, U

def project_gaussian(mu, Sigma, p, U):
    """
    将3D高斯分布投影到2D切平面 (支持批处理)
    输入:
        mu:    [N, 3]       - N个3D高斯分布的均值
        Sigma: [N, 3, 3]    - N个3D高斯分布的协方差矩阵
        p:     [B, 3]       - B个切点坐标
        U:     [B, 3, 2]    - B个切空间基矩阵
    
    返回:
        mu_2d:    [B, N, 2]     - 投影后的2D均值
        Sigma_2d: [B, N, 2, 2]  - 投影后的2D协方差
    """
    dtype = mu.dtype
    p = p.to(dtype)
    U = U.to(dtype)
    
    delta = mu.unsqueeze(0) - p.unsqueeze(1)  # [B, N, 3]

    
    
    mu_2d = torch.einsum('bji,bnj->bni', U, delta)  # [B, N, 2]
    
    U_transposed = U.transpose(-1, -2)  # [B, 2, 3]

    
    temp = torch.matmul(
        U_transposed.unsqueeze(1),  # [B, 1, 2, 3]
        Sigma.unsqueeze(0)          # [1, N, 3, 3]
    )  # 结果: [B, N, 2, 3]
    
    Sigma_2d = torch.matmul(
        temp,                        # [B, N, 2, 3]
        U.unsqueeze(1)               # [B, 1, 3, 2] -> 广播到 [B, N, 3, 2]
    )  # 结果: [B, N, 2, 2]
    return mu_2d, Sigma_2d

def dn_dtheta_fun(theta0,phi0):
    cos_theta = torch.cos(theta0)
    sin_theta = torch.sin(theta0)
    cos_phi = torch.cos(phi0)
    sin_phi = torch.sin(phi0)
    return torch.stack([
        cos_theta * cos_phi,
        cos_theta * sin_phi,
        -sin_theta
    ], dim=-1)

def dn_dphi_fun(theta0,phi0):
    sin_theta = torch.sin(theta0)
    cos_phi = torch.cos(phi0)
    sin_phi = torch.sin(phi0)
    return torch.stack([
        -sin_theta * sin_phi,
        sin_theta * cos_phi,
        torch.zeros_like(sin_phi)
    ], dim=-1) 

def Jacobian(U,dn_dthetas,dn_dphis,n0):
    temp1=torch.matmul(torch.matmul(n0.unsqueeze(-1),dn_dthetas.unsqueeze(-2)),n0.unsqueeze(-1)).squeeze()
    temp2=torch.matmul(torch.matmul(n0.unsqueeze(-1),dn_dphis.unsqueeze(-2)),n0.unsqueeze(-1)).squeeze()
    
    temp=torch.stack([dn_dthetas-temp1,dn_dphis-temp2],axis=-1)
    U_transposed = U.transpose(-1, -2)
    # pdb.set_trace()
    J=torch.matmul(U_transposed,temp)
    return J

# def rec_corners(J,theta_res,phi_res):
#     delta_theta=theta_res/2
#     delta_phi=phi_res/2
#     A1=torch.tensor([-delta_theta,-delta_phi],device=J.device)
#     B1=torch.tensor([delta_theta,delta_phi],device=J.device)
#     # pdb.set_trace()
#     aa=torch.einsum('bii,i->bi', J, A1)
#     bb=torch.einsum('bii,i->bi', J, B1)
#     x1,x2=aa[:,0:1],aa[:,1:]
#     y1,y2=bb[:,0:1],bb[:,1:]
    
#     return x1,x2,y1,y2



if __name__ == "__main__":
    B,N = 100,2000 ###B the number of angles;N the number of scatterers
    device = "cuda:0"
    np.random.seed(42)
    mus = np.random.uniform(-2, 2, (N, 3))
    def random_cov():
        A = np.random.randn(3,3)
        return A @ A.T + np.eye(3)*0.3
    Sigmas = np.array([random_cov() for _ in range(N)])

    phi_res, theta_res = 91, 72
    # phi_res, theta_res = 10, 10
    B=phi_res*theta_res

    thetas = (-63.5 + 90 + torch.linspace(90, -265, theta_res)) / 180 * np.pi
    phis = (-11.5 + torch.linspace(90, -90, phi_res)) / 180 * np.pi
    thetas = thetas.repeat(phi_res).to(device)
    phis =  phis.repeat_interleave(theta_res).to(device)

    BS = torch.zeros(3,device=device)
    mu=torch.tensor(mus,device=device)
    Sigma=torch.tensor(Sigmas,device=device)

    n0, u1, u2, U = tangent_basis(thetas, phis)
    mu_2d, Sigma_2d = project_gaussian(mu, Sigma, n0, U)

    print("2D均值形状:", mu_2d.shape)        # [4, 2]
    print("2D协方差形状:", Sigma_2d.shape)   # [4, 2, 2]

    dn_dphis=dn_dphi_fun(thetas,phis)
    dn_dthetas=dn_dtheta_fun(thetas,phis)
    print(dn_dthetas.shape)
    print(dn_dphis.shape)
    J=Jacobian(U,dn_dthetas,dn_dphis,n0)
    print('J shape:',J.shape)
    x1,x2,y1,y2=rec_corners(J,theta_res,phi_res)
    print('x1,x2,y1,y2 shape:',x1.shape)