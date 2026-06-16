 # -*- coding: utf-8 -*-
"""RadSplatter runner for training and testing
"""


import utils as utils
import loss_utils as loss_utils
from torch.cuda.amp import autocast, GradScaler
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import argparse
from shutil import copyfile
import numpy as np
import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from radsplatter_model import GaussModel 
from radsplatter_render import GaussRenderer
from datasets_aps_new import RSRP_dataset,RSRP_dataset_test
import time
import torch.nn as nn
import pdb
def find_max_A(A,max_number=3000):
    # pdb.set_trace()
    #A:[6552,32]
    matric=A.T
    norms = np.linalg.norm(matric, axis=1)
    # 排序并选择最大的900个元素
    max_indices = np.argsort(norms)[-max_number:]
    # pdb.set_trace()
    # 选择最大的900个元素
    return max_indices
class SmoothLoss(nn.Module):
    def __init__(self, lambda_tv=1.0):
        super(SmoothLoss, self).__init__()
        self.lambda_tv = lambda_tv
        self.mse_loss = nn.MSELoss()

    def forward(self, pred, target, coords):
        l2_loss = self.mse_loss(pred, target)
        tv_loss = self.total_variation(pred, coords)
        loss = l2_loss + self.lambda_tv * tv_loss
        return loss

    def total_variation(self, pred, coords):
        batch_size = coords.size(0)
        dist_matrix = torch.cdist(coords, coords)  
        knn_idx = dist_matrix.topk(2, largest=False).indices[:, 1]  
        # pdb.set_trace()
        diff = pred - pred[knn_idx]
        tv_loss = torch.mean(torch.norm(diff, p=1, dim=-1))
        return tv_loss

class DGS_Runner():
    def __init__(self,mode,world_size,num_scatters,num_max_angles,sh_up_iter,**kwargs):

        kwargs_path = kwargs['path']
#         kwargs_render = kwargs['render']
#         kwargs_network = kwargs['networks']
        kwargs_train = kwargs['train']

        kargs_newtrain=kwargs['newtrain']

        self.expname = kwargs_path['expname']

        if mode != 'test':
            self.batch_size = kwargs_train['batch_size']
        else:
            self.batch_size=1
            # self.batch_size = 128
        # self.datadir = kwargs_path['datadir']
        self.logdir = kwargs_path['logdir']
        self.sh_up_iter=sh_up_iter
        self.devices = torch.device('cuda')
  
       
        self.location_path='./data/location.txt' ######shanghai data/
     
        self.points_path='./data/pc_Vector_downsampled_points_final.txt'
        self.matrix_A_path='./data/Matrix_A_before.npy'


# 
        self.bs=torch.tensor([-204., -59., 22.10000038]).float().to(self.devices)

        self.Matrix_A = torch.tensor(np.load(self.matrix_A_path)).float()
       
        self.A_max_indice = find_max_A(self.Matrix_A,num_max_angles)

        self.world_size=world_size
        # self.bs=self.bs/self.world_size
        self.train_index_path='./data/train_index_ex.txt'
        self.test_index_path='./data/test_index_ex.txt'
        # self.test_index_path='/home/rorschach/yihengwang/radsplatter_code/data/train_index_ex.txt'
        print('=================================================================')
        print("Loading training set...")
        self.train_set = RSRP_dataset(self.train_index_path,torch.tensor([-204., -59., 22.10000038]).float() ,self.world_size)
        print("Loading test set...")
        self.test_set = RSRP_dataset(self.test_index_path,torch.tensor([-204., -59., 22.10000038]).float(), self.world_size)
        self.train_iter = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=0)
        if mode != 'test':
            self.test_iter = DataLoader(self.test_set, batch_size=self.batch_size//2, shuffle=False, num_workers=0)
        else:
            self.test_iter = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=0)
        print('=================================================================')
        print('=================================================================')
        print(f"Train set size:%d, Test set size:%d", len(self.train_set), len(self.test_set))
        print('=================================================================')
        self.points=(torch.tensor(np.loadtxt(self.points_path)).float().to(self.devices)-self.bs[None,:])/self.world_size
        self.points_=(torch.tensor(np.loadtxt(self.points_path)).float().to(self.devices)-self.bs[None,:])/self.world_size
        # pdb.set_trace()
        theta_res=91
        phi_res=72
        self.bs=torch.tensor([0,0,0]).float().to(self.devices)
        # self.gaussianmodel=GaussModel(P_BS=self.bs,theta_res=theta_res,phi_res=phi_res,mode=mode,angle_indice=self.A_max_indice).to(self.devices)
        self.gaussianmodel=GaussModel(world_size=self.world_size,P_BS=self.bs,theta_res=theta_res,phi_res=phi_res,mode=mode,angle_indice=self.A_max_indice).to(self.devices)
        self.M=num_scatters
        self.gaussianmodel.create_from_pcd(self.points,self.M,self.bs)

        self.eps = 0
        # pdb.set_trace()

        print('=================================================================')
        for name, param in self.gaussianmodel.named_parameters():
            if param.requires_grad:
                print(f"Updating {name} with size {param.size()}")
        print('=================================================================')
        # pdb.set_trace()
        self.epoch_per_evaluation=len(self.train_iter)*5

        

        self.scaler = GradScaler()

        self.optimizer=self.gaussianmodel.training_setup(kargs_newtrain)

        
        # params = list(self.gaussianmodel.parameters())
        # params=[]
        # for param in self.gaussianmodel.parameters():
            # params.append(param)
        # params=self.gaussianmodel.parameters()
# 
        # self.optimizer = torch.optim.Adam(params, lr=float(kwargs_train['lr']),
                                        #   weight_decay=float(kwargs_train['weight_decay']),
                                        #   betas=(0.9, 0.999))
        # self.cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer,
                                                                        # T_max=float(kwargs_train['T_max']), eta_min=float(kwargs_train['eta_min']),
                                                                        # last_epoch=-1)
        # pdb.set_trace()

        self.GaussRenderer=GaussRenderer(self.bs)
        

        self.current_iteration = 0
        if kwargs_train['load_ckpt'] or mode == 'test':
            # self.load_checkpoints()
            self.load_best_checkpoints()  
        self.batch_size = kwargs_train['batch_size']
        self.total_iterations = kwargs_train['total_iterations']
        self.save_freq = kwargs_train['save_freq']

    def load_checkpoints(self):
      
        ckptsdir = os.path.join(self.logdir, self.expname, 'ckpts_ex_2k_aps')
        if not os.path.exists(ckptsdir):
            os.makedirs(ckptsdir)
        ckpts = [os.path.join(ckptsdir, f) for f in sorted(os.listdir(ckptsdir)) if 'tar' in f]
        print('Found ckpts %s', ckpts)


        if len(ckpts) > 0:
            ckpt_path = ckpts[-1]
            print('Loading ckpt %s', ckpt_path)
            ckpt = torch.load(ckpt_path, map_location=self.devices)

            self.gaussianmodel.load_state_dict(ckpt['gaussianmodel_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            # self.cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer,T_max=20,eta_min=1e-5)
            # self.cosine_scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            self.current_iteration = ckpt['current_iteration']
        
    def save_checkpoint(self):
        ckptsdir = os.path.join(self.logdir, self.expname, 'ckpts_ex_2k_aps')
        model_lst = [x for x in sorted(os.listdir(ckptsdir)) if x.endswith('.tar')]
        if len(model_lst) > 2:
            print(model_lst)
            os.remove(ckptsdir + '/%s' % model_lst[0])

        ckptname = os.path.join(ckptsdir, '{:06d}.tar'.format(self.current_iteration))
        torch.save({
            'current_iteration': self.current_iteration,
            'gaussianmodel_state_dict': self.gaussianmodel.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            # 'scheduler_state_dict': self.cosine_scheduler.state_dict()
        }, ckptname)
        return ckptname
    def save_best_checkpoint(self,degree):
        ckptsdir = os.path.join(self.logdir, self.expname, 'ckpts_ex_2k_aps')
        model_lst = [x for x in sorted(os.listdir(ckptsdir)) if x.endswith('.pth')]
        if len(model_lst) > 0:
            print(model_lst)
            os.remove(ckptsdir + '/%s' % model_lst[0])

        ckptname = os.path.join(ckptsdir, 'best_model_{:06d}_SH_{:d}.pth'.format(self.current_iteration,degree))
        torch.save({
            'current_iteration': self.current_iteration,
            'gaussianmodel_state_dict': self.gaussianmodel.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            # 'scheduler_state_dict': self.cosine_scheduler.state_dict()
        }, ckptname)
        return ckptname
    def load_best_checkpoints(self):
        ckptsdir = os.path.join(self.logdir, self.expname, 'ckpts_ex_2k_aps')
        if not os.path.exists(ckptsdir):
            os.makedirs(ckptsdir)
        ckpts = [os.path.join(ckptsdir, f) for f in sorted(os.listdir(ckptsdir)) if 'pth' in f]
        print('Found ckpts %s', ckpts)

        if len(ckpts) > 0:
            ckpt_path = ckpts[-1]
            print('Loading ckpt %s', ckpt_path)
            ckpt = torch.load(ckpt_path, map_location=self.devices)

            self.gaussianmodel.load_state_dict(ckpt['gaussianmodel_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            # self.cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer,T_max=20,eta_min=1e-5)
            # self.cosine_scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            self.current_iteration = ckpt['current_iteration']

    def eval_network_rsrp(self):
        MAE=0
        ii=0
        times=[]
        flag=0
        self.gaussianmodel.eval()
        mae_bad_positions=[]
        predict_rsrp_db=[]
        ll=0
        indx=[]
        pp=0
        for test_input, test_label in tqdm(self.test_iter):
            # bactch_size=test_input.shape[0]
            
            # ii+=1
            position_grid=test_input.to(self.devices)
            # self.gaussianmodel.eval()
            t1=time.time()
            out1,pc,pc_,bias,T,TT=self.GaussRenderer(model=self.gaussianmodel,position_grids=position_grid,angle_indice=self.A_max_indice,eval=True)
            # out1,pc,pc_,bias,_,_=self.GaussRenderer(model=self.gaussianmodel,position_grids=position_grid,eval=True)
            out=10*torch.log10(self.Matrix_A.cuda()@out1.T+self.eps).T
            # out=10*torch.log10(self.Matrix_A.cuda()@out1.T).T
            t2=time.time()
            t_gap=(t2-t1)*1000
            print('Cost Time(ms)',t_gap)
            # pdb.set_trace()
            
            # pdb.set_trace()
            times.append(t_gap)
            # rsrp_dB=out*100-180
            rsrp_dB=out
            rsrp_GT_dB=test_label
            # pdb.set_trace()
            mae=np.mean(abs(rsrp_dB.detach().cpu().numpy() - rsrp_GT_dB.detach().cpu().numpy()))
            # mae=np.mean(abs(rsrp_dB.detach().cpu().numpy() - rsrp_GT_dB.detach().cpu().numpy()),-1)
            # mae1=np.sum(mae)/bactch_size
            
            # if mae>=20:
            #     # ll+=1
            #     indx.append(ll)
            #     mae_bad_positions.append(mae)
            #     predict_rsrp_db.append(rsrp_dB.detach().cpu().numpy())
            #     # mae=9.5
            #     pp+=1
            #     ll+=1
            #     print('continue')
            #     continue
            # else:
            #     ll+=1
            #     ii+=1

            ll+=1
            ii+=1
            
                
                
                # bad_positions.append(test_input.numpy())
                # print(ll,np.array(bad_positions).shape)
            print('==================Evaluating MAE(dB):',mae)
            MAE+=mae
            # if flag==0:
            #     pc_save=pc_.detach().cpu().numpy()
            #     print(pc_save.shape)
            #     np.savetxt("/home/rorschach/yihengwang/radsplatter_code/log/radsplatter_gird_cuda5/pc2k_learned_points_new.txt",pc_save, fmt='%s')
            #     pc_save=pc.detach().cpu().numpy()
            #     print(pc_save.shape)
            #     np.savetxt("/home/rorschach/yihengwang/radsplatter_code/log/radsplatter_gird_cuda5/pc2k_learned_points_biased_new.txt",pc_save, fmt='%s')
            #     flag=1
        
        MAE_mean=MAE/ii
        # MAE_mean=np.sum(MAE)/(ii*bactch_size)
        print('==================================')
     
        print('Total Test Mean MAE(dB):',MAE_mean)
        print('Total Test Mean Time(ms):',np.mean(times))
  
        print('==================================')
        return MAE_mean


    def train(self):
    
        Losss=[]
        MAE=[]
        best_loss=1e10
        bestb_loss=1e10
        maeb_loss= 1e10
        patient_i=0
        mae_= 1e10
        flag=0
        indictor=0
        sh_degree=4
        # self.gaussianmodel.train()
        while self.current_iteration <= self.total_iterations:
            loss_saved=0
            mae_saved=0
            # index=self.train_index
            # for id in tqdm(index):
            ll=0
            l=0
            
            for train_input, train_label in tqdm(self.train_iter):
                # pdb.set_trace()
                flag=flag+1

                ll+=self.batch_size
                l+=1
                position_grid=train_input.to(self.devices)
                self.gaussianmodel.train()
                
                # rsrp_GT=10**(train_label/10).to(self.devices)
                # rsrp_GT=10**(train_label/10).to(self.devices)*1e12
                rsrp_GT=10**(train_label/10).to(self.devices)
                self.optimizer.zero_grad()
                # pdb.set_trace()
                
                out1,pcs,pcs_,bias,T,TT=self.GaussRenderer(model=self.gaussianmodel,position_grids=position_grid,angle_indice=self.A_max_indice)

                # out1,pcs,pcs_,bias,T,TT,shs_real,shs_imag ,scales,rotations ,gamma1,gamma2,opacity,phi_o=self.GaussRenderer(model=self.gaussianmodel,position_grids=position_grid,angle_indice=self.A_max_indice)
               
                out=(self.Matrix_A.cuda()@out1.T+self.eps).T
                # out=(self.Matrix_A.cuda()@out1.T).T

                # pdb.set_trace()

           
                # l1=loss_utils.l1_loss(out1[:,self.A_max_indice],0)
                bs_regulariztion_term=torch.mean(torch.norm(pcs_-0,dim=1, keepdim=True))
                # bias_regulariztion_term=torch.norm(bias, 2)
                bias_regulariztion_term=torch.mean(torch.norm(bias, dim=1))
                # pdb.set_trace()
                # selection_regularization_term1=torch.norm(T.sum(dim=1) - 1)*10
                term4=torch.norm(T.max(1).values-1)

                # T_sparse_term=torch.norm(T,1)
                # selection_regularization_term2=torch.norm(torch.sum(T) - self.M)
                # selection_regularization_term3=torch.mean((T - 0.5)**2)
                # selection_regularization_term3=torch.norm(T - 0.5)
                # pdb.set_trace()
                criterion = SmoothLoss(lambda_tv=0.001)
                mse_loss = nn.MSELoss()
                # rsrp_dB_normalized=10*torch.log10(self.Matrix_A.cuda()@out1.T).T

                # rsrp_dB=rsrp_dB_normalized*100-180
                # rsrp_dB=10*torch.log10(self.Matrix_A.cuda()@out1.T).T

                rsrp_dB=10*torch.log10(self.Matrix_A.cuda()@out1.T+self.eps).T
                # rsrp_dB=out*120-180
                rsrp_GT_dB=train_label.cuda()
                MAE_loss=torch.mean(abs(rsrp_dB - rsrp_GT_dB))
                # rsrp_GT_dB_normalized=(train_label.cuda()+180)/100

                # mae_loss=nn.MAELoss()
                # loss = criterion(out, rsrp_GT, position_grid)*2+0.001*bs_regulariztion_term+term4*5+bias_regulariztion_term*0.01+l1*0.2
             
                # loss=MAE_loss*5+l1+mse_loss(out, rsrp_GT)*5+0.001*bs_regulariztion_term+term4*10+bias_regulariztion_term*0.01

                # loss=MAE_loss*5+criterion(out, rsrp_GT, position_grid)*5+0.001*bs_regulariztion_term+term4*20+bias_regulariztion_term*0.01
                # loss=MAE_loss*5+criterion(out, rsrp_GT, position_grid)*10+0.001*bs_regulariztion_term+term4*20+bias_regulariztion_term*0.01
                loss=MAE_loss*5+mse_loss(rsrp_dB, rsrp_GT_dB)*5+criterion(out, rsrp_GT, position_grid)*10+0.001*bs_regulariztion_term+term4*35+bias_regulariztion_term*0.001

                # loss=MAE_loss*5+mse_loss(rsrp_dB, rsrp_GT_dB)*5+0.001*bs_regulariztion_term+term4*25+bias_regulariztion_term*0.001
                # loss=criterion(out, rsrp_GT, position_grid)*20+0.001*bs_regulariztion_term+term4*25+bias_regulariztion_term*0.001
                # loss=criterion(out, rsrp_GT, position_grid)*15+0.001*bs_regulariztion_term+term4*20+bias_regulariztion_term*0.01
                total_loss=loss
                # print(out[0].mean())
                # print('MSE Loss',mse_loss(out, rsrp_GT))
                # print('MSE Loss',mse_loss(rsrp_dB, rsrp_GT_dB))
                # print('MSE Loss',mse_loss(rsrp_dB_normalized, rsrp_GT_dB_normalized))
                # print('MAE_loss',MAE_loss)
                print('Total Loss',total_loss)

                # Check for NaN/Inf in loss (early detection)
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print(f"============================================================")
                    print(f"ERROR: NaN or Inf detected in loss at iteration {self.current_iteration}")
                    print(f"Loss value: {total_loss}")
                    print(f"Saving emergency checkpoint...")
                    print(f"============================================================")
                    self.save_checkpoint()
                    raise ValueError(f"Training diverged: NaN/Inf loss at iteration {self.current_iteration}")

                # FIXED: Correct gradient computation order
                # 1. Zero gradients FIRST
                self.optimizer.zero_grad()

                # 2. Backward pass with gradient scaling
                self.scaler.scale(total_loss).backward()

                # 3. Unscale gradients before clipping (required for GradScaler)
                self.scaler.unscale_(self.optimizer)

                # 4. Clip gradients to prevent explosion
                torch.nn.utils.clip_grad_norm_(self.gaussianmodel.parameters(), max_norm=1.0)

                # 5. Update parameters ONCE
                self.scaler.step(self.optimizer)
                self.scaler.update()
      
                print('====================')
                ###check backpropagation
                # print('1',total_loss.grad)
                # print('2',T.grad.shape)
                # print('3',sum(sum(T.grad)))
                # print('4',sum(sum(TT.grad)))
                # print('5',bias)
                # print('6',bias.grad)
                # print('7',shs_real-shs_real.grad)
                # print('7',sum(shs_real.grad))
                # print('8',shs_imag)
                # print('8',sum(shs_imag.grad))
                # print('9',scales)
                # print('9',scales.grad)
                # print('10',rotations)
                # print('10',rotations.grad)
                # print('11',gamma1)
                # print('11',gamma1.grad)
                # print('12',gamma2)
                # print('12',gamma2.grad)
                # print('13',opacity)
                # print('13',opacity.grad)
                # print('14',phi_o)
                # print('14',phi_o.grad)
                # print('====================')
       
                # torch.nn.utils.clip_grad_norm_(self.gaussianmodel.parameters(), max_norm=1.0)
                self.current_iteration += 1
      
                # Every 1000 its we increase the levels of SH up to a maximum degree
                # if self.current_iteration % 1000 == 0:
                if self.current_iteration % self.sh_up_iter == 0:
                    if sh_degree<4:
                        sh_degree+=1
                    self.gaussianmodel.oneupSHdegree()
             
                loss_saved+=total_loss
                print('SH Degree=',sh_degree)

        
                mae=np.mean(abs(rsrp_dB.detach().cpu().numpy() - rsrp_GT_dB.detach().cpu().numpy()))
                print('MAE(dB)',mae)
                mae_saved+=mae
    
                
            loss_saved=loss_saved/l
            print('+++++++++++++++++++++++++++++++++')
            print('Total Mean LOSS:',loss_saved.detach().cpu().numpy())
            Losss.append(loss_saved.detach().cpu().numpy())
            # np.save('/data-test/user1/radsplatter/Loss/loss_0814_new.npy',Losss)
            # np.save('/data-test/user1/3dgs4lscm/batch_loss_mlp_pc6k_extra_up_regularization_regularization_sh2_.npy',Losss)
            mae_saved=mae_saved/l
            MAE.append(mae_saved)
            print('Total MAE(dB):',mae_saved)
            print('+++++++++++++++++++++++++++++++++')
        


            if loss_saved>=bestb_loss:
                patient_i+=1
                #  print('Early Stop')F
                #  break
                if patient_i>=50:
                     print('Early Stop')
                     break
            temp_=False
            
            
            # if loss_saved<bestb_loss:
            #         if patient_i>0:
            #              patient_i=0
            #         temp_=True
            #         bestb_loss=loss_saved
            #         ckptname = self.save_checkpoint()
            #         print('Save at'+ ckptname)
            if mae_saved<maeb_loss and temp_==False:
                    if patient_i>0:
                         patient_i=0
                    maeb_loss=mae_saved
                    ckptname = self.save_checkpoint()
                    print('Save at'+ ckptname)
            
            # if flag==self.epoch_per_evaluation:
            # if self.current_iteration>=2500:
            if self.current_iteration>=0:
                flag1=True
            else:
                flag1=(flag==self.epoch_per_evaluation)
            if flag1:
                flag=0
                m=self.eval_network_rsrp()
                if m<=mae_:
                    if indictor>0:
                        indictor=0
                    mae_=m
                    ckptname = self.save_best_checkpoint(sh_degree)
                    print('Save at'+ ckptname)
                else:
                    print('Continue')
                    # if self.current_iteration>=6000:
                    # if self.current_iteration>=3500:
                    if self.current_iteration>=0:
                        indictor+=1
                        if indictor>=50:
                            print('Overfitting STOP at:',self.current_iteration)
                            break
            



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='./radsplatter_setting_new.yml', help='config file path')
    parser.add_argument('--gpu', type=int, default=4)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--num_scatters',  type=int, default=2000) ###2500
    parser.add_argument('--world_size',  type=int, default=1)
    parser.add_argument('--num_max_angles',  type=int, default=800) ###800
    parser.add_argument('--sh_up_iter',  type=int, default=500) 

    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)

    with open(args.config) as f:
        kwargs = yaml.safe_load(f)
        f.close()

    worker = DGS_Runner(mode=args.mode,world_size=args.world_size,num_scatters=args.num_scatters,num_max_angles=args.num_max_angles,sh_up_iter=args.sh_up_iter, **kwargs)
    if args.mode == 'train':
        worker.train()
    elif args.mode == 'test':
        worker.eval_network_rsrp()
# 