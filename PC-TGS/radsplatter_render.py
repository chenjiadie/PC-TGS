import pdb
import torch
import torch.nn as nn
import math
# from einops import reduce
from complex_sh_utils_new import eval_sh
from projection_utils import *
# from mvnorm_2d_normal_cdf_utils import cdf_value
from pdf_utils import *
import torch.autograd.profiler as profiler
from prune_utils import filter_by_mahalanobis
USE_PROFILE = False
import contextlib
import pdb
from tqdm import tqdm

def nan_to_num_complex(x, nan=0.0, posinf=1e12, neginf=-1e12):
    return torch.complex(
        torch.nan_to_num(x.real, nan=nan, posinf=posinf, neginf=neginf),
        torch.nan_to_num(x.imag, nan=nan, posinf=posinf, neginf=neginf)
    )
class Embedder():
    """positional encoding
    """
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']    # input dimension of gamma
        out_dim = 0

        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']    # L-1, 10-1 by default
        N_freqs = self.kwargs['num_freqs']         # L


        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)  #2^[0,1,...,L-1]
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        """return: gamma(input)
        """
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)




def get_embedder(multires, is_embeded=True, input_dims=1):
    """get positional encoding function

    Parameters
    ----------
    multires : log2 of max freq for positional encoding, i.e., (L-1)
    i : set 1 for default positional encoding, 0 for none
    input_dims : input dimension of gamma


    Returns
    -------
        embedding function; output_dims
    """
    if is_embeded == False:
        return nn.Identity(), input_dims

    embed_kwargs = {
                'include_input' : True,
                'input_dims' : input_dims,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])+0.0001

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    if torch.any(torch.isnan(x)):
        print(r)
        print(q)
        pdb.set_trace()

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R
def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    if torch.any(torch.isnan(L)):
        print(s)
        pdb.set_trace()
    return L
def build_covariance_3d(s, r):
    L = build_scaling_rotation(s, r)
    actual_covariance = L @ L.transpose(1, 2)
    if torch.any(torch.isnan(actual_covariance)):
        pdb.set_trace()
    return actual_covariance
@torch.no_grad()
def get_radius(cov2d):
    det = cov2d[:, 0, 0] * cov2d[:,1,1] - cov2d[:, 0, 1] * cov2d[:,1,0]
    mid = 0.5 * (cov2d[:, 0,0] + cov2d[:,1,1])
    lambda1 = mid + torch.sqrt((mid**2-det).clip(min=0.1))
    lambda2 = mid - torch.sqrt((mid**2-det).clip(min=0.1))
    return 3.0 * torch.sqrt(torch.max(lambda1, lambda2)).ceil()

@torch.no_grad()
def get_rect(pix_coord, radii, width, height):
    rect_min = (pix_coord - radii[:,None])
    rect_max = (pix_coord + radii[:,None])
    rect_min[..., 0] = rect_min[..., 0].clip(0, width - 1.0)
    rect_min[..., 1] = rect_min[..., 1].clip(0, height - 1.0)
    rect_max[..., 0] = rect_max[..., 0].clip(0, width - 1.0)
    rect_max[..., 1] = rect_max[..., 1].clip(0, height - 1.0)
    return rect_min, rect_max
class GaussRenderer(nn.Module):
    """
    A gaussian splatting renderer

    >>> gaussModel = GaussModel.create_from_pcd(pts)
    >>> gaussRender = GaussRenderer()
    >>> out = gaussRender(pc=gaussModel)
    """

    def __init__(self, bs_location,white_bkgd=True, **kwargs):
        super(GaussRenderer, self).__init__()
      
        # self.mode
        # self.active_sh_degree = model.active_sh_degree
        self.debug = False
        self.white_bkgd = white_bkgd
        self.bs=bs_location
        # pdb.set_trace()
        self.device= bs_location.device
        print('-----------------------------------------------------------------')
        print('RadSplatter Render Intialization on:', self.device)
        print('-----------------------------------------------------------------')
        self.embed_depth_bs_fn, self.input_depth_bs_dim = get_embedder(10, True, 3)
        print('=================================================================')
        print('BS Position :',  self.bs)
        print('=================================================================')
   
        

    # def build_S(self, means3D, shs_real,shs_imag, degree, Grid_posiotion):
    #     rays_o2 = Grid_posiotion
    #     rays_d2 =  rays_o2[:,None,:]-means3D[None,:,:] 
    #     rays_d_normarlized2=rays_d2/rays_d2.norm(dim=2,keepdim=True)
    #     # rays_d_normarlized=rays_d/rays_d.norm(dim=1,keepdim=True)
    #     # pdb.set_trace()
    #     Complex_S = eval_sh(degree, shs_real.permute(0,2,1),shs_imag.permute(0,2,1), rays_d_normarlized2)
    #     return Complex_S

    def build_S(self, means3D, shs_real,shs_imag, degree, BS_position,Grid_posiotion):
        rays_o1 = BS_position
        rays_d1 = means3D[None,:,:] - rays_o1[None,None,:]
        rays_d_normarlized1=rays_d1/rays_d1.norm(dim=2,keepdim=True)
        rays_o2 = Grid_posiotion
        rays_d2 =  rays_o2[:,None,:]-means3D[None,:,:] 
        rays_d_normarlized2=rays_d2/rays_d2.norm(dim=2,keepdim=True)
        Complex_S = eval_sh(degree, shs_real.permute(0,2,1),shs_imag.permute(0,2,1), rays_d_normarlized1,rays_d_normarlized2)
        return Complex_S
    
    
    
    def render(self,index,L,weight,Complex_S,Alpha):
      
        # temp1=Alpha[None,None,:,:]*weight[None,:,:,None]*L[:,None,:,:]
        
        num_grid=index.shape[0]
        num_scatterer=index.shape[1]
        # pdb.set_trace()
        num_angles=weight.shape[0]

        
        # index_L= index.expand(-1, 3)
        sorted_L=torch.gather(L, 1, index)

        index_2=index.expand(num_grid,num_scatterer,num_angles)
        # pdb.set_trace()

        # if Complex_S.shape[0]!=num_grid:
            
        #     Complex_S_=Complex_S.unsqueeze(0).expand(num_grid,-1,-1).cuda()
        # else:
        #     print('SH Degree>=1')
        #     # Complex_S_=Complex_S
        # pdb.set_trace()
        sorted_Complex_S=torch.gather(Complex_S.expand(num_grid,num_scatterer,num_angles), 1, index_2)
        # sorted_Complex_S=torch.gather(Complex_S.expand(num_grid,num_scatterer,1), 1, index)

        sorted_Alpha=torch.gather(Alpha.unsqueeze(0).expand(num_grid,num_scatterer,1),1,index)

        sorted_weight=torch.gather(weight.unsqueeze(0).expand(num_grid,num_angles,num_scatterer).permute(0,2,1),1, index_2)

        # mask=(sorted_weight!=0).float()
        # pdb.set_trace()
        # sorted_weight1=sorted_weight*mask/
        # mask = torch.sigmoid(20 * (sorted_weight - 1e-10))
        # soft_w=sorted_weight*mask

        # alpha=sorted_Alpha*sorted_weight*sorted_L

        # alpha=sorted_Alpha*soft_w

        # temp1=alpha*sorted_L

        temp1=sorted_Alpha*sorted_weight*sorted_L
        
        T=torch.cat([torch.ones_like(temp1[:,:1,:]),(1-temp1[:,:-1,:])],dim=1).cumprod(dim=1)
        # pdb.set_trace()

        alpha=sorted_Alpha*sorted_weight
     
        render_S=(T * alpha * sorted_Complex_S).sum(dim=1)
        

        
        render_aps=torch.abs(render_S)**2
        # render_aps = torch.exp(render_aps) 

        # render_aps_clamped = torch.clamp(render_aps, min=1e-10, max=1.0)

        if torch.any(torch.isnan(render_aps)):
            pdb.set_trace()
        return   render_aps

    # def forward(self, position_grid, model, **kwargs):
    def forward(self, model,position_grids,angle_indice,eval=False, **kwargs):
        batchsize, _ = position_grids.shape
        # opacity,phi_o = model.get_opacity
        linear1 = model.get_opacity
        scales = model.get_scaling
        rotations = model.get_rotation
        shs_real,shs_imag = model.get_features
        gamma1,gamma2=model.get_gamma
        # M=shs_real.shape[0]
        n0,U,areaA,S_shift=model.get_project
        # def expan(x,M):
        #     x_old=x
        #     x_new=x.unsqueeze(1).expand(x_old.shape[0],M,x_old.shape[1])
        #     return x_new
        
        
        # bias = model.get_bias
        if eval:
            T,TT_middle=model.get_selection_matrix_eval
            means3d_middle,means3D,bias= model.get_xyz_eval
        else:
            T,TT_middle=model.get_selection_matrix
            means3d_middle,means3D,bias= model.get_xyz
        
        if USE_PROFILE:
            prof = profiler.record_function
        else:
            prof = contextlib.nullcontext

        direction_BS=self.bs[None,:]-means3D
        
        depths_to_BS=torch.norm(direction_BS, dim=1, keepdim=True)
        # depths_to_BS=torch.norm(direction, dim=1)
        # sorted_depths_to_BS, index = torch.sort(depths_to_BS,dim=0)
        depths_to_BS_EB=self.embed_depth_bs_fn(depths_to_BS)
        out1=linear1(depths_to_BS_EB)
        # pdb.set_trace()
        opacity=out1[:,0:1]
        phi_o=out1[:,1:]
        # pdb.set_trace()

        ##old best
        # out1=linear1(sorted_depths_to_BS_EB)
        # sorted_opacity=out1[:,0:1]
        # sorted_phi_o=out1[:,1:]


        # index_mean= index.expand(-1, 3)
        # sorted_means3D=torch.gather(means3D, 0, index_mean)

        depths_to_grid=torch.norm(position_grids[:,None,:]-means3D[None,:,:],dim=2, keepdim=True)
        _, index = torch.sort(depths_to_grid,dim=1)
        # depths_to_grid_EB=self.embed_depth_bs_fn(depths_to_grid)
        # pdb.set_trace()
        # out1=linear1(depths_to_grid_EB)
        # sorted_opacity=out1[:,:,0:1]
        # sorted_phi_o=out1[:,:,1:]


   
        # index_shs=index.unsqueeze(-1).expand(-1, 25,1)
        # sorted_shs_real=torch.gather(shs_real,0,index_shs)
        # sorted_shs_imag=torch.gather(shs_imag,0,index_shs)
        # index_scales=index.expand(-1, 3)
        # index_rotations=index.expand(-1, 4)
        # sorted_scales=torch.gather(scales,0,index_scales)
        # sorted_rotations=torch.gather(rotations,0,index_rotations)


        # direction_grid=position_grids[:,None,:]-sorted_means3D[None,:,:]
        # sorted_depths_to_grid=torch.norm(direction_grid,dim=2, keepdim=True)


        # pdb.set_trace()
        
       
        with prof("build color"):
            Complex_S = self.build_S(means3D=means3D, shs_real=shs_real, shs_imag=shs_imag, degree=model.active_sh_degree,  BS_position=self.bs,Grid_posiotion=position_grids)
            # Complex_S = self.build_S(means3D=means3D, shs_real=shs_real, shs_imag=shs_imag, degree=model.active_sh_degree,  Grid_posiotion=position_grids)

        # opacity=opacity_*torch.exp(1j * phi_o)

        
        with prof("build cov3d"):
            cov3D = build_covariance_3d(scales, rotations) #the number of cloud point*3*3
            # pdb.set_trace()
           
            means2D, cov2D = project_gaussian(means3D, cov3D, n0.to(self.device), U.to(self.device))
            # pdb.set_trace()
            # mu2D, cov2D=filter_by_mahalanobis(sorted_means2D, sorted_cov2D)

        with prof("build explcitly geometric path loss"):
            # L=1/(((600*sorted_depths_to_BS[None,:,:])**gamma1)*((600*sorted_depths_to_grid)**gamma2)+1e-10)
            L=1/(((depths_to_BS[None,:,:])**gamma1[None,:,:])*((depths_to_grid)**gamma2[None,:,:])+1e-10)
            # L=1/(((depths_to_BS[None,:,:])**2)*((depths_to_grid)**2)+1e-10)

        
        
        # pdb.set_trace()
        recv_signal = torch.zeros(batchsize,  6552).cuda()
        chunks = 800  # 100
        chunks_num = angle_indice.shape[0] // chunks
        # for i in tqdm(range(chunks_num)):
        for i in range(chunks_num):
            means2d=means2D[i*chunks:(i+1)*chunks]
            cov2d=cov2D[i*chunks:(i+1)*chunks]
            s_shift=S_shift[i*chunks:(i+1)*chunks].to(self.device)
            areaa=areaA[i*chunks:(i+1)*chunks].to(self.device)
            weight=(areaa[:,None]*multivariate_normal_pdf_origin(means2d,cov2d+s_shift[:,None,:]))

            # weight=(areaa[:,None]*multivariate_normal_pdf_stable(means2d,cov2d+s_shift[:,None,:]))
            # for i in range(means2d.shape[0]):
            #     for j in
            
            
            # pdb.set_trace()
            # sorted_weight=(areaa[:,None]*multivariate_normal_pdf_origin(sorted_means2d,sorted_cov2d))
            # sorted_weight=(areaa[:,None]*multivariate_normal_pdf_stable(sorted_means2d,sorted_cov2d+s_shift[:,None,:]))
            
            # sorted_weight=(areaa[:,None]*multivariate_normal_pdf_robust(sorted_means2d,sorted_cov2d+s_shift[:,None,:]))
            # weight1=filter_by_mahalanobis(means2d, cov2d,weight)
            # pdb.set_trace()

            if torch.any(torch.isnan(means2d)) or torch.any(torch.isnan(cov2d)):
                pdb.set_trace()
            # if torch.min(weight1)<0:
                # print(torch.min(weight1))
                # pdb.set_trace()
            if torch.any(torch.isnan(weight)):
                pdb.set_trace()
            
            # pdb.set_trace()

        
            with prof("render"):
                recv_signal_chunks = self.render(
                index=index,
                L=L, 
                # weight=weight,
                weight=weight,
                Complex_S=Complex_S,
                Alpha=opacity*torch.exp(1j*phi_o)
                )
            recv_signal[..., angle_indice[i * chunks:(i + 1) * chunks]]=recv_signal_chunks
        # pdb.set_trace()
      
        return recv_signal,means3D,means3d_middle,bias,T,TT_middle
        # return recv_signal,means3D,means3d_middle,bias,T,TT_middle,shs_real,shs_imag ,scales,rotations ,gamma1,gamma2,opacity,phi_o