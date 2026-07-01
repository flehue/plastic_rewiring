#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 16:06:24 2025

@author: elida
"""


import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


n_blocks = 20    
size_block = 12  
sizes = [size_block] * n_blocks 
# p_intra = 0.80     # prob. conexión dentro del bloque
# p_inter = 0.02     # prob. conexión entre bloques


# p = [[p_intra if i == j else p_inter for j in range(n_blocks)]
#      for i in range(n_blocks)]
rng = np.random.default_rng(42)


p_intra_vals = rng.uniform(0.5, 1, size=n_blocks)             
p_inter_vals = rng.uniform(0.1, 1, size=(n_blocks, n_blocks)) 


p = [[p_intra_vals[i] if i == j else p_inter_vals[i, j] for j in range(n_blocks)]
     for i in range(n_blocks)]


p = np.array(p)
p = (p + p.T) / 2.0 


p = p.tolist()

G = nx.stochastic_block_model(sizes, p, seed=42)

density = nx.density(G)


avg_clustering = nx.average_clustering(G, weight='weight')


degrees = np.array([val for (node, val) in G.degree(weight=None)])  # usa weight=None para grado simple
mean_degree = degrees.mean()
std_degree = degrees.std()


print("Densidad global:         ", f"{density:.4f}")
print(" Clustering promedio:     ", f"{avg_clustering:.4f}")
print("Grado medio:             ", f"{mean_degree:.4f}")
print("Desviación estándar:     ", f"{std_degree:.4f}")


A = nx.to_numpy_array(G, dtype=float) 

rng = np.random.default_rng(42)
A[A > 0] = rng.uniform(0, 50, size=(A > 0).sum())

A = np.triu(A, 1)
A = A + A.T
np.fill_diagonal(A, 0)


D = np.zeros_like(A, dtype=float)
for i in range(A.shape[0]):
    bi = i // size_block  
    for j in range(i+1, A.shape[1]):
       
        bj = j // size_block
        if bi == bj:
      
            D_ij = rng.uniform(1, 14)
        else:
      
            D_ij = rng.uniform(15.0, 30)
        D[i, j] = D_ij
        D[j, i] = D_ij

D *= (A > 0)


plt.figure(figsize=(6, 6))
plt.imshow(A, cmap='viridis', interpolation='nearest')
plt.title('Matriz de pesos (0–50) — 240 nodos, 20 bloques de 12')
plt.xlabel('Nodos'); plt.ylabel('Nodos')
plt.colorbar(label='Peso')
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 6))
plt.imshow(D, cmap='viridis', interpolation='nearest')
plt.title('Matriz de delays (ms) — intra < inter')
plt.xlabel('Nodos'); plt.ylabel('Nodos')
plt.colorbar(label='Delay (ms)')
plt.tight_layout()
plt.show()


from scipy.io import savemat
data_dict = {'W': A, 'D': D}
# savemat('matrices_modulares_240_densida_het_10blocks.mat', data_dict)



