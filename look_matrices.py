folder = "toy_model_elida/"
import sys
sys.path.append(folder)
import Toy_model_v_final_okok as toy_model
import rewire

import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt


P_block = loadmat(folder+"P_block.mat")
label_count = np.array([(i,(P_block['Labels']==i).sum()) for i in range(1,21)])


#%%load the very matrices

matrix_data = loadmat(folder+"matrices_syn_seed_1002_con ajuste.mat")
C,D = matrix_data["W_syn"],matrix_data["D_syn"]

dens_syn, clust_syn, mean_deg_syn, std_deg_syn = toy_model.compute_global_graph_metrics(C)

print("MÉTRICAS SINTÉTICAS INICIALES ")
print(f"Densidad:            {dens_syn:.4f}")
print(f"Clustering promedio: {clust_syn:.4f}")
print(f"Grado medio:         {mean_deg_syn:.2f}")
print(f"Desv. grado:         {std_deg_syn:.2f}")
    
    
    
    
    


#%%

plt.figure(1)
plt.clf()

plt.subplot(221)
plt.title("SC mat")
plt.imshow(C,cmap="jet")
plt.colorbar()

plt.subplot(222)
plt.title("delay mat")
plt.imshow(D,cmap="jet")
plt.colorbar()

plt.subplot(212)
plt.title("nodes per module")
plt.bar(label_count[:,0],label_count[:,1])

plt.tight_layout()
plt.show()


