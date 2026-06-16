import torch
import torch.nn  as nn
import numpy as np
import math
import pdb
from utils import build_scaling_rotation, inverse_sigmoid,strip_symmetric
# from complex_sh_utils import RGB2SH
from projection_utils import *
# from mvnorm_2d_normal_cdf_utils import cdf_value
from scipy.spatial import KDTree
from torch.autograd import Function


def row_max_to_one(matrix):
    max_values, _ = torch.max(matrix, dim=1, keepdim=True)
    result = (matrix == max_values)
    return result.float()



    
def distCUDA2(points):
    points_np = points.detach().cpu().float().numpy()
    dists, inds = KDTree(points_np).query(points_np, k=4)
    meanDists = (dists[:, 1:] ** 2).mean(1)

    meanDists = np.clip(meanDists, 1e-8, np.inf)

    return torch.tensor(meanDists, dtype=points.dtype, device=points.device)
def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper
class GaussModel(nn.Module):
    """
    A Gaussian Model

    * Attributes
    _feature_dc_real: DC term of features (\tau_real)
    _feature_rest_real: rest features (\tau_real)
    _feature_dc_imag: DC term of features (\tau_imag)
    _feature_rest_imag: rest features (\tau_imag)
    _rotatoin: rotation of gaussians
    _scaling: scaling of gaussians
    _T: Selection Matrix of RM Scheme
    _bias: Bias Term of RM Scheme
    _opacity: opacity of gaussians

    >>> gaussModel = GaussModel.create_from_pcd(pts)
    >>> gaussRender = GaussRenderer()
    >>> out = gaussRender(pc=gaussModel, camera=camera)
    """
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid

        self.bias_activation = torch.tanh
        # self.bias_activation =torch.relu
        # self.T_activation = torch.sigmoid
        self.inverse_T_activation = inverse_sigmoid

        self.T_activation = nn.Softmax(dim=1)

        self.inverse_opacity_activation = inverse_sigmoid
        self.projection_activation=torch.sigmoid

        self.rotation_activation = torch.nn.functional.normalize
    
    # def __init__(self, P_BS,theta_res,phi_res,angle_indice,sh_degree : int=4,debug=False,mode='train'):
    def __init__(self,world_size, P_BS,theta_res,phi_res,angle_indice,sh_degree : int=4,debug=False,mode='train'):

        super(GaussModel, self).__init__()
        print('-----------------------------------------------------------------')
        print('RadSplatter Model Intialization on World Size:', world_size)
        print('-----------------------------------------------------------------')
        if mode=='train':
            self.active_sh_degree=0
        elif mode=='test':
            print('==Test==')
            self.active_sh_degree=sh_degree ####test set =4 train =0
        self.max_sh_degree = sh_degree 
        print('=================================================================')
        print("The maximum degree of complex SH function : ", self.max_sh_degree)
        print("The total modes of complex SH function :",(self.max_sh_degree+1)**2) 
        print('=================================================================')
        self.angle_num=len(angle_indice)
        print('=================================================================')
        print('The recovered maximum bins of angles in APSs :', self.angle_num)
        print('=================================================================')
    
        self._features_dc_real = torch.empty(0)
        self._features_rest_real = torch.empty(0)

        self._features_dc_imag = torch.empty(0)
        self._features_rest_imag = torch.empty(0)


        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)

        self._T = torch.empty(0)
        self._bias = torch.empty(0)

        theta = (-63.5 + 90 + torch.linspace(90, -265, theta_res)) / 180 * np.pi
        phi = (-11.5 + torch.linspace(90, -90, phi_res)) / 180 * np.pi

        thetas = theta.repeat(phi_res)
        phis =  phi.repeat_interleave(theta_res)
        thetas_res=(theta[0]-theta[1])/2
        phis_res=(phi[0]-phi[1])/2
        print('=================================================================')
        print('Resolution of Theta (Rad):',thetas_res)
        print('Resolution of Phi (Rad):',phis_res)
        print('=================================================================')

        # self.thetas_res=(theta[0]-theta[1])/2
        # self.phis_res=(phi[0]-phi[1])/2
     
        # self.thetas=thetas[angle_indice]
        # self.phis=phis[angle_indice]
        self.thetas=thetas[angle_indice]
        self.phis=phis[angle_indice]
        # pdb.set_trace()

        # self.n0, _, _, self.U=tangent_basis(P_BS.cpu(),thetas, phis)
        # dn_dphis=dn_dphi_fun(thetas,phis)
        # dn_dthetas=dn_dtheta_fun(thetas,phis)
        n0, _, _, U=tangent_basis(P_BS.cpu(),thetas, phis)
        self.n0=n0*0.3/world_size
        self.U=U*0.3/world_size
        dn_dphis=dn_dphi_fun(thetas,phis)*0.3/world_size
        dn_dthetas=dn_dtheta_fun(thetas,phis)*0.3/world_size
        # self.n0=n0/world_size
        # # self.U=U/world_size
        # # dn_dphis=dn_dphi_fun(thetas,phis)/world_size
        # # dn_dthetas=dn_dtheta_fun(thetas,phis)/world_size
        print('=================================================================')
        print("dn_dthetas shape:",dn_dthetas.shape)
        print("dn_dphis shape:",dn_dphis.shape)
        J=Jacobian(self.U,dn_dthetas,dn_dphis,self.n0)
        print('J shape:',J.shape)
     
        # # self.x1,self.x2,self.y1,self.y2=rec_corners(J,thetas_res,phis_res)
        # # print('Corners position (x1,x2,y1,y2) shape:',self.x1.shape)
        self.areaA=4* thetas_res* phis_res*torch.abs(torch.linalg.det(J))
        print('Area A shape:', self.areaA.shape)
        S_ang=torch.tensor([[thetas_res**2/3,0],[0, phis_res**2/3]])
        # print(S_ang)
        self.S_shift=torch.matmul(torch.matmul(J,S_ang[None,:,:]),J.transpose(-1, -2))
        # pdb.set_trace()
        print('S shape:', self.S_shift.shape)
        # pdb.set_trace()
        # self.linear1=nn.Sequential(
        #                 nn.Linear(21,30),
        #                 # nn.ReLU(),
        #                 nn.Sigmoid(),
        #                 nn.BatchNorm1d(30),
        #                 nn.Linear(30,15),
        #                 # nn.ReLU(),
        #                 nn.Sigmoid(),
        #                 nn.BatchNorm1d(15),
        #                 nn.Linear(15,2),
        #                 # nn.ReLU()
        #                 nn.Sigmoid()
        # )

        self.linear1=nn.Sequential(
                        nn.Linear(21,30),
                        # nn.ReLU(),
                        nn.Sigmoid(),
                        nn.LayerNorm(30),
                        nn.Linear(30,15),
                        # nn.ReLU(),
                        nn.Sigmoid(),
                        nn.LayerNorm(15),
                        nn.Linear(15,2),
                        # nn.ReLU()
                        nn.Sigmoid()
        )
        # self.linear2=nn.Sequential(
        #                 nn.Linear(63,30),
        #                 nn.ReLU(),
        #                 # nn.Sigmoid(),
        #                 nn.BatchNorm1d(30),
        #                 nn.Linear(30,15),
        #                 nn.ReLU(),
        #                 # nn.Sigmoid(),
        #                 nn.BatchNorm1d(15),
        #                 nn.Linear(15,1),
        #                 nn.ReLU()
        #                 # nn.Sigmoid()
        # )
        print('=================================================================')
     
        
        self.setup_functions()
        self.debug = debug


    def create_from_pcd(self, pcd,M,P_BS):
        """
            create the guassian model from raw point cloud
            pcd: raw point cloud
            M: the number of virtual scatterers
            P_BS: the position of BS
        """
        points = pcd #the number of points*3 (x,y,z), tensor
        fused_point_cloud= points.float().cuda()
        N=pcd.shape[0]
        print('=================================================================')
        print("Number of raw points cloud at initialisation : ", N)
        print("Number of virtual scatterers after selection : ", M)
        print('=================================================================')
        # 
        distances = torch.norm(fused_point_cloud - P_BS, dim=1)  # Compute distances to the base station
        # _, indices = torch.topk(-distances, M) 
        indices=torch.randperm(N)[:M]

        # T_init_ = torch.zeros(M, N)+1/(N)
        T_init_ = torch.zeros(M, N)
        T_init_[torch.arange(M), indices] = 1

        # T_init=inverse_sigmoid(T_init_)
        T_init=T_init_
        b_init=torch.zeros(M,3)

        # gamma1_init=torch.zeros(M,1)
        # gamma2_init=torch.zeros(M,1)

        gamma1_init=torch.ones(M,1)*2
        gamma2_init=torch.ones(M,1)*2

        # bias1_init=torch.zeros()


        # phi_init=torch.zeros(M,1)

        # phi_o_init=torch.zeros(M,1)

        # S_real= torch.ones(M, 1) *0.5
        # S_imag= torch.ones(M, 1) *0.5

        S_real= torch.ones(M, self.angle_num) *0.5
        S_imag= torch.ones(M, self.angle_num) *0.5

  
        # fused_S_real=  RGB2SH(S_real.float().cuda())
        # fused_S_imag=  RGB2SH(S_imag.float().cuda())
        fused_S_real=  S_real.float().cuda()
        fused_S_imag=  S_imag.float().cuda()
        
        features_real = torch.zeros((fused_S_real.shape[0], self.angle_num, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features_real[:, :self.angle_num, 0 ] = fused_S_real
        features_real[:, self.angle_num:, 1:] = 0.0

        features_imag = torch.zeros((fused_S_imag.shape[0], self.angle_num, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features_imag[:, :self.angle_num, 0 ] = fused_S_imag
        features_imag[:, self.angle_num:, 1:] = 0.0

        # features_real = torch.zeros((fused_S_real.shape[0], 1, (self.max_sh_degree + 1) ** 2)).float().cuda()
        # features_real[:, :1, 0 ] = fused_S_real
        # features_real[:, 1:, 1:] = 0.0

        # features_imag = torch.zeros((fused_S_imag.shape[0], 1, (self.max_sh_degree + 1) ** 2)).float().cuda()
        # features_imag[:, :1, 0 ] = fused_S_imag
        # features_imag[:, 1:, 1:] = 0.0

        # dist2 =torch.clamp_min(torch.rand(fused_point_cloud.shape[0]).float().cuda(),0.0000001)
        point=points[indices,:].detach().cpu().numpy()
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(point)).float().cuda()), 0.0001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((M, 4), device="cuda")
        # rots=torch.ones((M, 4), device="cuda")
        rots[:, 0] = 1

        # opacities = inverse_sigmoid(0.1 * torch.ones((M, 1), dtype=torch.float, device="cuda"))

        
        self._init_xyz=fused_point_cloud

        self._T=nn.Parameter((T_init).contiguous().float().cuda().requires_grad_(True))
        self._b=nn.Parameter(b_init.contiguous().float().cuda().requires_grad_(True))
        self._features_dc_real = nn.Parameter(features_real[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest_real = nn.Parameter(features_real[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_dc_imag= nn.Parameter(features_imag[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest_imag = nn.Parameter(features_imag[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        # self._opacity = nn.Parameter(opacities.requires_grad_(True))
        # self._phi_o=nn.Parameter(phi_o_init.contiguous().float().cuda().requires_grad_(True))
        self._gamma1=nn.Parameter((gamma1_init).contiguous().float().cuda().requires_grad_(True))
        self._gamma2=nn.Parameter((gamma2_init).contiguous().float().cuda().requires_grad_(True))

        # self._bias2=nn.Parameter((gamma2_init).contiguous().float().cuda().requires_grad_(True))
       

        return self

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)+0.000001
    

    @property
    def get_bias(self):
        # return self.bias_activation(self._b),self._b
        return self._b
        # return self.bias_activation(self._b)
   
    @property
    def get_selection_matrix(self):
        selection_matrix =self.T_activation(self._T)
        return selection_matrix, self._T
    
  
    
    @property
    def get_selection_matrix_eval(self):
        selection_matrix =self.T_activation(self._T)
        selection_matrix_=row_max_to_one(selection_matrix)
        # pdb.set_trace()
        # selection_matrix_ = torch.where(selection_matrix > 0.5, torch.tensor(1.0,device=selection_matrix.device), torch.tensor(0.0,device=selection_matrix.device))
        return selection_matrix_, self._T

    @property
    def get_xyz(self):
        T,_=self.get_selection_matrix
        # ll=800
        b=self.get_bias
        _xyz=T@self._init_xyz
        # print(T[:,0])
        # pdb.set_trace()

        _xyz_=_xyz.detach().clone()
        _xyz_[:,2]= torch.maximum(_xyz_[:, 2], torch.tensor(0.0))
        # _xyz_[:,0]= torch.maximum(_xyz_[:, 2], torch.tensor(-ll))
        # _xyz_[:,1]= torch.maximum(_xyz_[:, 2], torch.tensor(-ll))
        # _xyz_[:,0]= torch.min(_xyz_[:, 2], torch.tensor(ll))
        # _xyz_[:,1]= torch.min(_xyz_[:, 2], torch.tensor(ll))
        _xyz_bias=  T@self._init_xyz+b
        # _xyz_bias=  T@self._init_xyz
        _xyz_bias_=_xyz_bias.detach().clone()
        _xyz_bias_[:,2]= torch.maximum(_xyz_bias[:, 2], torch.tensor(0.0))
        # _xyz_bias_[:,0]= torch.maximum(_xyz_bias[:, 2], torch.tensor(-ll))
        # _xyz_bias_[:,1]= torch.maximum(_xyz_bias[:, 2], torch.tensor(-ll))
        # _xyz_bias_[:,0]= torch.min(_xyz_bias[:, 2], torch.tensor(ll))
        # _xyz_bias_[:,1]= torch.min(_xyz_bias[:, 2], torch.tensor(ll))
        return _xyz, _xyz_bias,b
    
    @property
    def get_xyz_eval(self):
        T,_=self.get_selection_matrix_eval
        # ll=800
        b=self.get_bias
        _xyz=T@self._init_xyz
        _xyz_=_xyz.detach().clone()
        _xyz_[:,2]= torch.maximum(_xyz_[:, 2], torch.tensor(0.0))
        # _xyz_[:,0]= torch.maximum(_xyz_[:, 2], torch.tensor(-ll))
        # _xyz_[:,1]= torch.maximum(_xyz_[:, 2], torch.tensor(-ll))
        # _xyz_[:,0]= torch.min(_xyz_[:, 2], torch.tensor(ll))
        # _xyz_[:,1]= torch.min(_xyz_[:, 2], torch.tensor(ll))
        _xyz_bias=  T@self._init_xyz+b
        # _xyz_bias=  T@self._init_xyz
        _xyz_bias_=_xyz_bias.detach().clone()
        _xyz_bias_[:,2]= torch.maximum(_xyz_bias[:, 2], torch.tensor(0.0))
        # _xyz_bias_[:,0]= torch.maximum(_xyz_bias[:, 2], torch.tensor(-ll))
        # _xyz_bias_[:,1]= torch.maximum(_xyz_bias[:, 2], torch.tensor(-ll))
        # _xyz_bias_[:,0]= torch.min(_xyz_bias[:, 2], torch.tensor(ll))
        # _xyz_bias_[:,1]= torch.min(_xyz_bias[:, 2], torch.tensor(ll))
        return _xyz, _xyz_bias,b
    
    @property
    def get_features(self):
        features_dc_real = self._features_dc_real
        features_rest_real = self._features_rest_real
        features_real=torch.cat((features_dc_real, features_rest_real), dim=1)

        features_dc_imag = self._features_dc_imag
        features_rest_imag = self._features_rest_imag
        features_imag=torch.cat((features_dc_imag, features_rest_imag), dim=1)
        return features_real,features_imag
    
    @property
    # def get_opacity(self):
        # return self.opacity_activation(self._opacity),self._phi_o
    def get_opacity(self):
        # return self.opacity_activation(self._opacity),self._phi_o
        return self.linear1
    
    @property
    def get_gamma(self):
        # return self._project_a, self._project_b, self._project_c, self.projection_activation
        return  self._gamma1,self._gamma2
    @property
    def get_project(self):
        return self.n0,self.U,self.areaA,self.S_shift
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
    
    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            print('==========')
            print('SH degree plus 1')
            print('==========')
            self.active_sh_degree += 1
    
    def training_setup(self, training_args):

        l = [
            # {'params': [self._xyz], 'lr': training_args['position_lr_init'], "name": "xyz"},
            {'params': [self._features_dc_real], 'lr': training_args['feature_lr'], "name": "f_dc"},
            {'params': [self._features_rest_real], 'lr': training_args['feature_lr']/ 20.0, "name": "f_rest"},
            {'params': [self._features_dc_imag], 'lr': training_args['feature_lr'], "name": "f_dc"},
            {'params': [self._features_rest_imag], 'lr': training_args['feature_lr']/ 20.0, "name": "f_rest"},
            # {'params': [self._opacity], 'lr': training_args['opacity_lr'], "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args['scaling_lr'], "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args['rotation_lr'], "name": "rotation"},
            {'params': self.linear1.parameters(), 'lr': training_args['linear1_lr'], "name": "linear1"},
            {'params': [self._T], 'lr': training_args['T_lr'], "name": "selection"},
            {'params': [self._b], 'lr': training_args['bias_lr'], "name": "bias"},
            # {'params': [self._phi_o], 'lr': training_args['opacity_lr'], "name": "phi"},
            {'params': [self._gamma1], 'lr': training_args['gamma_lr'], "name": "gamma1"},
            {'params': [self._gamma2], 'lr': training_args['gamma_lr'], "name": "gamma2"},
 
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args['position_lr_init'],
                                                    lr_final=training_args['position_lr_final'],
                                                    lr_delay_mult=training_args['position_lr_delay_mult'],
                                                    max_steps=training_args['position_lr_max_steps'])
        return  self.optimizer 
 

