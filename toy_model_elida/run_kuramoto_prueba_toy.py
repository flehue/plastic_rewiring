#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 19:56:14 2025

@author: bdl
"""

import numpy as np
import scipy.io as sio
from KuramotoClassFor import Kuramoto
from scipy.io import loadmat
import concurrent.futures
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import os
t1=time.time()

# archivo_dist='/home/bdl/Descargas/matriz_distancias_toy_250.mat'
# archivo_conect='/home/bdl/Descargas/matriz_conectividad_toy_250.mat'

import scipy.io as sio

data = loadmat('/home/elida/Descargas/matrices_syn_seed_1002_sin ajuste.mat')


# data_conect = sio.loadmat(archivo_conect)
# data_dist = sio.loadmat(archivo_dist)

df_c = data["W_syn"]
df_d=data['D_syn']


plt.figure()
plt.imshow(df_c, cmap="viridis")
plt.colorbar(label="Weights")  # barra de color

plt.figure()
plt.imshow(df_d, cmap="viridis")
plt.colorbar(label="Delays")  # barra de color


# plt.show()


struct_connectivity = df_c.copy() 
delays_matrix = df_d.copy()
# delays_matrix/= 1000 

N = struct_connectivity .shape[0]
struct_connectivity[np.diag(np.ones(N))==0] /= struct_connectivity[np.diag(np.ones(N))==0].mean()

print(struct_connectivity .shape)
print(delays_matrix.shape)

dt = 1e-3
simulation_period = 60


# natfreqs = np.full(N, 40)

# nat_freq_std = 0
seed = 335

# MD_values =[0.017,0.023,0.028]#,0.021,0.030

# K_values=[350,400]
# MD_values = np.arange(30, 35, 1)  # de 0 a 30 en pasos de 1


# MD_values=MD_values *1e-3

# K_values = np.arange(450, 650, 50)  # de 100 a 1100 en pasos de 50
# # # # # K_values=[N]
K_values=[400]


MD_values =[0.024]#0.033,

# K_values=[0.5*N,1*N,1.5*N,2*N,2.5*N,3*N,3.5*N,4*N]


directory = '../output_timeseries/'

def run_simulation(MD_K):
    MD, K = MD_K
    print(f"Comenzando simulación para K={K}, MD={MD}")
    
     
    filename = directory + f'toy_N{N}_K{K:.3F}_MD{MD:.3f}_seed{seed}.mat'
 
    # model = Kuramoto(struct_connectivity=struct_connectivity, delays_matrix=delays_matrix,K=K
    #                  ,dt=dt, simulation_period=simulation_period,n_nodes=N, 
    #                  natfreqs=natfreqs, nat_freq_std=nat_freq_std, SEED=seed,mean_delay=MD)
    
    
    model = Kuramoto(struct_connectivity=struct_connectivity, delays_matrix=delays_matrix,K=K
                      ,dt=dt, simulation_period=simulation_period,n_nodes=N, 
                      natfreqs=None,nat_freq_mean=40, nat_freq_std=2,GenerateRandom=True,SEED=seed,mean_delay=MD)
 
   
    
    
    
    R, dynamics = model.simulate()

  
    data = {'theta': dynamics, 'kop': R}
    sio.savemat(filename, data)
    print(f"Simulación terminada para K={K}, MD={MD}")


tasks = [(MD, K) for MD in MD_values for K in K_values]

# run_simulation(tasks)
max_workers_numb =1
with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers_numb) as executor:
    executor.map(run_simulation, tasks)

t2=time.time()

print('simulation time',t2-t1)
