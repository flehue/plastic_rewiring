#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 4 13:11:25 2025

@author: elida
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.io import savemat


n_blocks   = 20       




all_blocks = np.arange(n_blocks)
p_intra_vals = rng.uniform(0.7, 1,n_blocks) 


p = rng.uniform(0.5, 1, size=(n_blocks, n_blocks))


p = (p + p.T) / 2.0

for b in range(n_blocks):
    p[b, b] = p_intra_vals[b]


G = nx.stochastic_block_model(size_block, p, seed=42)

density        = nx.density(G)
avg_clustering = nx.average_clustering(G)
degrees        = np.array([deg for _, deg in G.degree()])
mean_degree    = degrees.mean()
std_degree     = degrees.std()

print("Densidad global:         ", f"{density:.4f}")
print("Clustering promedio:     ", f"{avg_clustering:.4f}")
print("Grado medio:             ", f"{mean_degree:.4f}")
print("Desviación estándar:     ", f"{std_degree:.4f}")


A_bin = nx.to_numpy_array(G, dtype=int)  # 1 si hay arista, 0 si no


W = np.zeros_like(A_bin, dtype=float)

mask_edges = A_bin > 0
bi = block_index[:, None]
bj = block_index[None, :]

intra_mask = mask_edges & (bi == bj)
inter_mask = mask_edges & (bi != bj)

# Asignar pesos
# W[intra_mask] = rng.uniform(100.0, 200, size=intra_mask.sum())
# W[inter_mask] = rng.uniform(0.0, 50,  size=inter_mask.sum())

mu_intra, sigma_intra = 100, 800  # media y desviación para intra-bloque
mu_inter, sigma_inter = 10, 20    # media y desviación para inter-bloque

# # --- Generar pesos ---
# W[intra_mask] = rng.normal(mu_intra, sigma_intra, size=intra_mask.sum())
# W[inter_mask] = rng.normal(mu_inter, sigma_inter, size=inter_mask.sum())

def truncated_normal(mu, sigma, size, lower=0, upper=None):
    """Genera una muestra de una normal truncada entre [lower, upper]."""
    vals = rng.normal(mu, sigma, size)
    while True:
        # Si hay límite superior definido
        if upper is not None:
            bad = (vals < lower) | (vals > upper)
        else:
            bad = vals < lower
        # Si no hay valores fuera de rango, terminamos
        if not np.any(bad):
            break
        # Re-muestreamos solo los valores fuera del rango
        vals[bad] = rng.normal(mu, sigma, bad.sum())
    return vals


W[intra_mask] = truncated_normal(mu_intra, sigma_intra, intra_mask.sum(), lower=0)
W[inter_mask] = truncated_normal(mu_inter, sigma_inter, inter_mask.sum(), lower=0)


W = np.triu(W, 1)
W = W + W.T
np.fill_diagonal(W, 0.0)


# D = np.zeros_like(W, dtype=float)

# for i in range(N):
#     for j in range(i + 1, N):
#         if W[i, j] == 0:
#             continue
#         if block_index[i] == block_index[j]:
#             delay = rng.uniform(1.0, 14.0)
#         else:
#             delay = rng.uniform(15.0, 30.0)
#         D[i, j] = delay
#         D[j, i] = delay


mean_w_intra = W[intra_mask].mean() if intra_mask.any() else 0
mean_w_inter = W[inter_mask].mean() if inter_mask.any() else 0

def threshold_to_target_density(W, density_target, density_tol=None):
    """
    Ajusta la densidad de W (y D coherentemente) para aproximarse a density_target,
    conservando las aristas con mayor peso.

    Si density_tol no es None, y la densidad actual ya está dentro de
    [density_target - density_tol, density_target + density_tol],
    no se hace ningún cambio.
    """
    N = W.shape[0]
    num_possible = N * (N - 1) // 2

    A_bin = (W > 0).astype(int)
    M_current = A_bin.sum() // 2
    dens_current = M_current / num_possible

    if density_tol is not None:
        if abs(dens_current - density_target) <= density_tol:
            print(f"[threshold] Densidad actual {dens_current:.4f} dentro de tolerancia "
                  f"de objetivo {density_target:.4f} (tol={density_tol:.4f}).")
            return W.copy()#, D.copy() if D is not None else None

    M_target = int(round(density_target * num_possible))

    triu_idx = np.triu_indices(N, 1)
    weights = W[triu_idx]

    nonzero_mask = weights > 0
    nz_weights = weights[nonzero_mask]
    nz_indices = np.where(nonzero_mask)[0]

    if M_target >= len(nz_weights):
        return W.copy() #, D.copy() if D is not None else None

    order = np.argsort(-nz_weights)
    keep_idx = nz_indices[order[:M_target]]

    W_thr = np.zeros_like(W)
    # if D is not None:
    #     D_thr = np.zeros_like(D)
    # else:
    #     D_thr = None

    keep_rows = triu_idx[0][keep_idx]
    keep_cols = triu_idx[1][keep_idx]

    W_thr[keep_rows, keep_cols] = W[keep_rows, keep_cols]
    W_thr = W_thr + W_thr.T
    np.fill_diagonal(W_thr, 0.0)

    # if D_thr is not None:
    #     D_thr[keep_rows, keep_cols] = D[keep_rows, keep_cols]
    #     D_thr = D_thr + D_thr.T
    #     np.fill_diagonal(D_thr, 0.0)

    return W_thr

W_thr = threshold_to_target_density(
    W=W,

    density_target=0.54,
    density_tol=0.01   # tolerancia de ±0.01
)

def compute_global_graph_metrics(C):
    """
    A partir de una matriz de conectividad C (pesos),
    binariza con C>0 y calcula:
      - densidad
      - clustering promedio
      - grado medio
      - desviación estándar del grado
    """
    A_bin = (C > 0).astype(int)
    np.fill_diagonal(A_bin, 0)
    G = nx.from_numpy_array(A_bin)

    density = nx.density(G)
    avg_clustering = nx.average_clustering(G)
    degrees = np.array([d for _, d in G.degree()])
    mean_degree = degrees.mean()
    std_degree = degrees.std()

    return density, avg_clustering, mean_degree, std_degree

density, avg_clustering, mean_degree, std_degree=compute_global_graph_metrics(W_thr)
 
print('················································')
print("Densidad global:         ", f"{density:.4f}")
print("Clustering promedio:     ", f"{avg_clustering:.4f}")
print("Grado medio:             ", f"{mean_degree:.4f}")
print("Desviación estándar:     ", f"{std_degree:.4f}")
plt.figure(figsize=(6, 6))
plt.imshow(W_thr, cmap='viridis', interpolation='nearest')
plt.title(f'Matriz de pesos — {N} nodos, {n_blocks} bloques de {size_block}')
plt.xlabel('Nodos'); plt.ylabel('Nodos')
plt.colorbar(label='Peso',shrink=0.6)
plt.tight_layout()
plt.show()

# plt.figure(figsize=(6, 6))
# plt.imshow(D, cmap='viridis', interpolation='nearest')
# plt.title('Matriz de delays (ms) — intra < inter')
# plt.xlabel('Nodos'); plt.ylabel('Nodos')
# plt.colorbar(label='Delay (ms)')
# plt.tight_layout()
# plt.show()
