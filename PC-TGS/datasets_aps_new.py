import os
import random

# import imageio
import numpy as np
# import pandas as pd
import torch
import yaml
# from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from tqdm import tqdm

class RSRP_dataset(Dataset):

    def __init__(self, indexdir,BS, scale_worldsize=1):
      
        super().__init__()

        self.rsrpdata_dir='./data/RSRP_before.npy'
        self.location='./data/location.txt'

        self.dataset_index = np.loadtxt(indexdir)


        self.rx_poses = torch.from_numpy(np.loadtxt(
            self.location))  
        self.rx_poses = (self.rx_poses -BS[None,:])/ scale_worldsize

       
        self.RSRPs = torch.from_numpy(np.load(
            self.rsrpdata_dir))  



        self.nn_inputs, self.nn_RSRPs = self.load_data()

    def load_data(self):
        """load data from datadir to memory for training

        Returns
        --------
        nn_inputs : tensor. [n_samples, 3]. The inputs for training
                    position_grid:3

        nn_RSRPs : tensor. [n_samples, n_rsrp = 32]. The RSRP (dB) as labels
        """
        ## NOTE! Large dataset may cause OOM?
        nn_inputs = torch.tensor(np.zeros((len(self), 3)), dtype=torch.float32)
        nn_RSRPs = torch.tensor(np.zeros((len(self), 32)), dtype=torch.float32)


    
        data_counter = 0
        for idx in tqdm(self.dataset_index, total=len(self.dataset_index)):  # sample from dataset_index
            idx=int(idx)

         
            nn_inputs[data_counter] = self.rx_poses[idx]
            nn_RSRPs[data_counter] = self.RSRPs[idx]  
            data_counter += 1

        return nn_inputs, nn_RSRPs
      

    def __len__(self):
        return len(self.dataset_index)  

    def __getitem__(self, index):
        return self.nn_inputs[index], self.nn_RSRPs[index] 

class RSRP_dataset_test(Dataset):

    def __init__(self, indexdir,BS, scale_worldsize=1):
      
        super().__init__()

        self.rsrpdata_dir='./data/RSRP_after.npy'
        self.location='./data/location.txt'

        self.dataset_index = np.loadtxt(indexdir)


        self.rx_poses = torch.from_numpy(np.loadtxt(
            self.location))  
        self.rx_poses = (self.rx_poses -BS[None,:])/ scale_worldsize

       
        self.RSRPs = torch.from_numpy(np.load(
            self.rsrpdata_dir))  



        self.nn_inputs, self.nn_RSRPs = self.load_data()

    def load_data(self):
        """load data from datadir to memory for training

        Returns
        --------
        nn_inputs : tensor. [n_samples, 3]. The inputs for training
                    position_grid:3

        nn_RSRPs : tensor. [n_samples, n_rsrp = 32]. The RSRP (dB) as labels
        """
        ## NOTE! Large dataset may cause OOM?
        nn_inputs = torch.tensor(np.zeros((len(self), 3)), dtype=torch.float32)
        nn_RSRPs = torch.tensor(np.zeros((len(self), 32)), dtype=torch.float32)


    
        data_counter = 0
        for idx in tqdm(self.dataset_index, total=len(self.dataset_index)):  # sample from dataset_index
            idx=int(idx)

         
            nn_inputs[data_counter] = self.rx_poses[idx]
            nn_RSRPs[data_counter] = self.RSRPs[idx]  
            data_counter += 1

        return nn_inputs, nn_RSRPs
      

    def __len__(self):
        return len(self.dataset_index)  

    def __getitem__(self, index):
        return self.nn_inputs[index], self.nn_RSRPs[index] 


class RSRP_APS_dataset(Dataset):

    def __init__(self, indexdir,BS, scale_worldsize=1):
      
        super().__init__()
# _before_update_0420.npy
        self.rsrpdata_dir='./data/RSRP_after.npy'

        self.apsdata_dir='./data/angular_power_spectrum.npy'
        self.location='./data/location.txt'

        self.dataset_index = np.loadtxt(indexdir)


        self.rx_poses = torch.from_numpy(np.loadtxt(
            self.location))  
        self.rx_poses = (self.rx_poses -BS[None,:])/ scale_worldsize

       
        self.RSRPs = torch.from_numpy(np.load(
            self.rsrpdata_dir))  
        self.APSs = torch.from_numpy(np.load(
            self.apsdata_dir))


        self.nn_inputs, self.nn_RSRPs, self.nn_APSs = self.load_data()

    def load_data(self):
        """load data from datadir to memory for training

        Returns
        --------
        nn_inputs : tensor. [n_samples, 3]. The inputs for training
                    position_grid:3

        nn_RSRPs : tensor. [n_samples, n_rsrp = 32]. The RSRP (dB) as labels
        nn_APSs : tensor. [n_samples, n_aps = 6552]. The APS as labels
        """
        ## NOTE! Large dataset may cause OOM?
        nn_inputs = torch.tensor(np.zeros((len(self), 3)), dtype=torch.float32)
        nn_RSRPs = torch.tensor(np.zeros((len(self), 32)), dtype=torch.float32)
        nn_APSs = torch.tensor(np.zeros((len(self), 6552)), dtype=torch.float32)

    
        data_counter = 0
        for idx in tqdm(self.dataset_index, total=len(self.dataset_index)):  # sample from dataset_index
            idx=int(idx)

         
            nn_inputs[data_counter] = self.rx_poses[idx]
            nn_RSRPs[data_counter] = self.RSRPs[idx]
            nn_APSs[data_counter] = self.APSs[idx]  
            data_counter += 1

        return nn_inputs, nn_RSRPs,  nn_APSs,
      

    def __len__(self):
        return len(self.dataset_index)  

    def __getitem__(self, index):
        return self.nn_inputs[index], self.nn_RSRPs[index], self.nn_APSs[index]  
