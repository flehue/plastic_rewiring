import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import gc

import sys
sys.path.append(r"C:\Users\Fernando\Desktop\plasticity\Kuramoto_plasticity")
sys.path.append(r"C:\Users\Fernando\Desktop\plasticity\plastic_rewiring\toy_model_elida")
import Toy_model_v_final_okok as toy_model
import KuramotoClassFor as Kuramoto
import rewire



folder = "toy_model_elida/"
P_block = loadmat(folder+"P_block.mat")
label_count = np.array([(i,(P_block['Labels']==i).sum()) for i in range(1,21)])


#%%load matrices and original graph metrics
sc_file = "matrices_syn_seed_1002_con ajuste.mat"
matrix_data = loadmat(folder+sc_file)
C,D = matrix_data["W_syn"],matrix_data["D_syn"]

dens_syn, clust_syn, mean_deg_syn, std_deg_syn = toy_model.compute_global_graph_metrics(C)


#%% run model with the original matrix

N = D.shape[0]
dt = 1e-3
simulation_period = 60

seed = 335
K,MD = 400,0.024

model = Kuramoto(
        struct_connectivity=C,
        delays_matrix=D,
        K=K, dt=dt,
        simulation_period=simulation_period,
        n_nodes=N,
        natfreqs=None,
        nat_freq_mean=40, nat_freq_std=2,
        GenerateRandom=True, SEED=seed,
        mean_delay=MD
    )

R, dynamics = model.simulate()
np.savez_compressed(
    f"output/rewire/simulation_K{K:.3f}_MD{MD:.3f}.npz",
    kop=R, dynamics=dynamics,sc_file = sc_file
    )
gc.collect()


