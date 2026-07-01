# -*- coding: utf-8 -*-
"""
Created on Fri Dec 26 10:25:35 2025

@author: elida
"""


import numpy as np
import networkx as nx
from scipy.io import loadmat
import matplotlib.pyplot as plt

seed=1002

def truncated_normal(mu, sigma, size, lower=0.0, upper=None):
 
    rng = np.random.default_rng()
    vals = rng.normal(mu, sigma, size)
    while True:
        if upper is not None:
            bad = (vals < lower) | (vals > upper)
        else:
            bad = (vals < lower)
        if not np.any(bad):
            break
        vals[bad] = rng.normal(mu, sigma, bad.sum())
    return vals


def compute_global_graph_metrics(C):

    A_bin = (C > 0).astype(int)
    np.fill_diagonal(A_bin, 0)
    G = nx.from_numpy_array(A_bin)

    density = nx.density(G)
    avg_clustering = nx.average_clustering(G)
    degrees = np.array([d for _, d in G.degree()])
    mean_degree = degrees.mean()
    std_degree = degrees.std()

    return density, avg_clustering, mean_degree, std_degree


def build_sbm_adjacency_from_Pblock(labels, P_block, seed=seed):

    labels = np.asarray(labels).ravel()
    regs = np.unique(labels)
    region_indices = {r: np.where(labels == r)[0] for r in regs}
    sizes = [len(region_indices[r]) for r in regs]
    G_sbm = nx.stochastic_block_model(sizes, P_block, seed=seed)
    A_sbm = nx.to_numpy_array(G_sbm, dtype=int)
    return A_sbm


def enforce_target_clustering_on_A(A,
                                  desired_clust=0.74,
                                  rel_margin=0.01,
                                  max_passes=5,
                                  batch_checks=100,
                                  seed=None
                                  ):
    """
    Ajusta la matriz de adyacencia A añadiendo aristas que cierran triángulos,
    hasta que el coeficiente de clustering medio esté cerca de desired_clust.
    """

    rng = np.random.default_rng(seed)
    N = A.shape[0]
    A_new = A.copy().astype(bool)

    low  = desired_clust - rel_margin
    high = desired_clust + rel_margin

    for p in range(max_passes):
        G_tmp = nx.from_numpy_array(A_new.astype(int))
        clust = nx.average_clustering(G_tmp)
        if low <= clust <= high:
            break
        if clust > high:
            break

        closable_pairs = set() 

        for i in range(N):
            neigh = np.where(A_new[i])[0]  #obtener los vecinos del nodo
            k = len(neigh)
            if k < 2:
                continue

            
            for u_idx in range(k):  ## recorre todas las parejas de vecinos (u, v) de i
                for v_idx in range(u_idx + 1, k):
                    u = neigh[u_idx]
                    v = neigh[v_idx]

             
                    if A_new[u, v]: # Si ya existe la arista, no se hace nada pq ya esta cerrado el triangulo
                        continue

                    if u > v:
                        u, v = v, u  
                                            # Normalizar (u, v) para poder detectar duplicados despues (u, v) y (v, u)
                    closable_pairs.add((u, v))


        if not closable_pairs:
            break
        pair_list = list(closable_pairs)
        order = rng.permutation(len(pair_list)) # se añaden de manera aleatoria las aristas

        edges_added = 0
        for idx in order:
            u, v = pair_list[idx]

        
            if A_new[u, v]:     # Puede que la hayamos añadido ya en esta misma pasada
                continue
            A_new[u, v] = True
            A_new[v, u] = True
            edges_added += 1

           
            if edges_added % batch_checks == 0:  # Cada 'batch_checks' de aristas añadidas se recalcula el clustering
                G_tmp = nx.from_numpy_array(A_new.astype(int))
                clust = nx.average_clustering(G_tmp)

                if low <= clust <= high:
                    return A_new.astype(int)
                if clust > high:
                    return A_new.astype(int)

        if edges_added == 0:
            break

    return A_new.astype(int)


def enforce_target_density_on_A(A,
                                desired_density,
                                density_tol
                                ):

    A = A.copy().astype(int)
    N = A.shape[0]
    num_possible = N * (N - 1) // 2  #numero posible de aristas

    iu, iv = np.triu_indices(N, 1) 
    edges_mask = A[iu, iv] > 0
    M_current = edges_mask.sum()
    dens_current = M_current / num_possible  #se calcula la densidad (esto da lo mismo q usando la formulaxa de nx)

    if abs(dens_current - desired_density) <= density_tol:

        return A

    M_target = int(round(desired_density * num_possible))

    neighbors = {i: set(np.where(A[i])[0]) for i in range(N)}

    if M_current > M_target:     # caso en q es necesario eliminar aristas
        num_to_remove = M_current - M_target
        edge_indices = np.where(edges_mask)[0]

        if num_to_remove > 0 and len(edge_indices) > 0:
        
            tri_counts = []
            for idx in edge_indices:
                u = iu[idx]
                v = iv[idx]
                common = neighbors[u].intersection(neighbors[v])
                tri_counts.append(len(common))   # Para cada arista existente se ve en cuantos triangulos esta
            tri_counts = np.array(tri_counts)

         
            order = np.argsort(tri_counts)  # para sacar las q menos forman triangulos
            remove_idx = edge_indices[order[:num_to_remove]]

            ru = iu[remove_idx]
            rv = iv[remove_idx]
            A[ru, rv] = 0
            A[rv, ru] = 0

    
    elif M_current < M_target:  #no creo q sea necesario (caso en el q es necesario añadir aristas)
        num_to_add = M_target - M_current
        non_edges_mask = ~edges_mask
        nonedge_indices = np.where(non_edges_mask)[0]

        if num_to_add > 0 and len(nonedge_indices) > 0:
            tri_pot = []
            for idx in nonedge_indices:
                u = iu[idx]
                v = iv[idx]
          
                common = neighbors[u].intersection(neighbors[v])
                tri_pot.append(len(common))
            tri_pot = np.array(tri_pot)
            order = np.argsort(-tri_pot)  # de mayor a menor
            num_to_add = min(num_to_add, len(order))
            add_idx = nonedge_indices[order[:num_to_add]]

            au = iu[add_idx]
            av = iv[add_idx]
            A[au, av] = 1
            A[av, au] = 1

    return A


if __name__ == "__main__":
    size_block = [14,14,12,12,16,12,12,14,12,10,12,8,8,12,14,16,12,8,12,16]
    N = sum(size_block)
    
    P_block_mat = loadmat('P_block.mat')
    # P_block     = P_block_mat['P_block']
    labels     = P_block_mat['Labels']
    # labels = []
    # for region_id, count in enumerate(size_block, start=1):
    #     labels.extend([region_id] * count)
    # labels = np.array(labels)
    
    
    desired_density = 0.54      # densidad objetivo 
    desired_clust   = 0.74      # clustering objetivo
    density_tol     = 0.01      # tolerancia en densidad
    rel_margin      = 0.01      # banda relativa alrededor de desired_clust
    
    D_min = 0.001
    D_max = 0.031
    
    n_regs = len(size_block)
    
    # P_block_mat = loadmat('P_block.mat')
    # # P_block     = P_block_mat['P_block']
    # # # labels       = P_block_mat['Labels']
    # mu_block_W = P_block_mat['mu_block']
    # sigma_block_W=P_block_mat['sigma_block']
    # mu_block_D  =P_block_mat['D_mu_block']  
    # sigma_block_D=P_block_mat['D_sigma_block']   
    def truncated_normal_rng(mu, sigma, size, lower=0.0, upper=None, rng=None):
        """
        Igual idea que truncated_normal, pero usando un rng externo.
        """
        if rng is None:
            rng = np.random.default_rng()
        vals = rng.normal(mu, sigma, size)
        while True:
            if upper is not None:
                bad = (vals < lower) | (vals > upper)
            else:
                bad = (vals < lower)
            if not np.any(bad):
                break
            vals[bad] = rng.normal(mu, sigma, bad.sum())
        return vals
                            
    
    n_regs = len(size_block)
    
    
    # Rango aproximado observado
    intra_min = 0.7
    intra_max = 1.0
    inter_min = 0.1
    inter_max = 1.0
    
    def generate_P_block_uniform_ranges(
            n_regs,
            intra_min, intra_max,
            inter_min, inter_max,
            seed=None
        ):
        rng = np.random.default_rng(seed)
        P_block = np.zeros((n_regs, n_regs), dtype=float)
    
        for r in range(n_regs):
            for s in range(r, n_regs):
    
                if r == s:  # INTRA
                    val = rng.uniform(intra_min, intra_max)
                else:       # INTER
                    val = rng.uniform(inter_min, inter_max)
    
                P_block[r, s] = P_block[s, r] = val
    
        return P_block
    
    
    P_block = generate_P_block_uniform_ranges(
        n_regs=n_regs,
        intra_min=intra_min,
        intra_max=intra_max,
        inter_min=inter_min,
        inter_max=inter_max,
        seed=seed
    )
    
    A_syn = build_sbm_adjacency_from_Pblock(labels, P_block, seed=seed)
    
    dens_syn, clust_syn, mean_deg_syn, std_deg_syn = compute_global_graph_metrics(A_syn)
    print("MÉTRICAS SINTÉTICAS INICIALES ")
    print(f"Densidad:            {dens_syn:.4f}")
    print(f"Clustering promedio: {clust_syn:.4f}")
    print(f"Grado medio:         {mean_deg_syn:.2f}")
    print(f"Desv. grado:         {std_deg_syn:.2f}")
    
    
    A_tmp= enforce_target_clustering_on_A(
        A=A_syn,
        desired_clust=desired_clust,
        rel_margin=rel_margin,
        max_passes=5,
        batch_checks=100,
        seed=seed
    )
    
    A_final = enforce_target_density_on_A(
        A=A_tmp,
        desired_density=desired_density,
        density_tol=density_tol
    )
    
    
    
    dens_topo, clust_topo, mean_deg_topo, std_deg_topo = compute_global_graph_metrics(A_final)
    
    
    
    
    A_bin = (A_final > 0).astype(int)
    
    print("MÉTRICAS despues de aumentar densidad y clustering ")
    print(f"Densidad:            {dens_topo:.4f}")
    print(f"Clustering promedio: {clust_topo:.4f}")
    print(f"Grado medio:         {mean_deg_topo:.2f}")
    print(f"Desv. grado:         {std_deg_topo:.2f}")
    #%%
    def generate_connectome_with_given_adjacency(A,
                                                 labels,
                                                 mu_block,
                                                 sigma_block,
                                                 D_mu_block=None,
                                                 D_sigma_block=None,
                                                 D_min=None,
                                                 D_max=None,
                                                 W_min=0.0,
                                                 W_max=None,
                                                 seed=123,
                                                 weight_dist="normal_trunc",
                                                 delay_dist="normal_trunc"):
    
        rng = np.random.default_rng(seed)
        labels = np.asarray(labels).ravel()
        regs = np.unique(labels)
        reg_to_idx = {r: i for i, r in enumerate(regs)}
    
        N = A.shape[0]
        W = np.zeros((N, N), dtype=float)
        D = np.zeros((N, N), dtype=float) if D_mu_block is not None else None
    
        def sample_value(dist, mu, sigma):
            if sigma <= 0:
                return mu
    
            if dist == "normal_trunc":
                # truncada SOLO por abajo (0). El tope superior lo aplicamos fuera con clip.
                return truncated_normal(mu, sigma, 1, lower=0.0)[0]
    
            elif dist == "uniform":
                delta = np.sqrt(12) * sigma
                a = mu - delta/2
                b = mu + delta/2
                return rng.uniform(a, b)
    
            elif dist == "lognormal":
                # OJO: esto asume que mu/sigma son mean/std del espacio original (positivo)
                phi = np.sqrt(sigma**2 + mu**2)
                m = np.log(mu**2 / phi)
                s = np.sqrt(np.log(phi**2 / mu**2))
                return rng.lognormal(mean=m, sigma=s)
    
            elif dist == "gamma":
                # Evitar problemas si mu muy pequeño
                mu = max(mu, 1e-12)
                sigma = max(sigma, 1e-12)
                shape = (mu / sigma)**2
                scale = sigma**2 / mu
                return rng.gamma(shape, scale)
    
            else:
                raise ValueError(f"Unknown dist: {dist}")
    
        for i in range(N):
            ri = reg_to_idx[labels[i]]
            for j in range(i+1, N):
                if A[i, j] == 0:
                    continue
                rj = reg_to_idx[labels[j]]
    
                # ===== PESOS =====
                mu_w  = mu_block[ri, rj]
                sig_w = sigma_block[ri, rj]
                w = sample_value(weight_dist, mu_w, sig_w)
    
                # aplicar límites
                if W_min is not None:
                    w = max(W_min, w)
                if W_max is not None:
                    w = min(W_max, w)
    
                W[i, j] = W[j, i] = w
    
            
                if D is not None:
                    mu_d  = D_mu_block[ri, rj]
                    sig_d = D_sigma_block[ri, rj]
                    d = sample_value(delay_dist, mu_d, sig_d)
    
                    if D_min is not None:
                        d = max(D_min, d)
                    if D_max is not None:
                        d = min(D_max, d)
    
                    D[i, j] = D[j, i] = d
    
        np.fill_diagonal(W, 0.0)
        if D is not None:
            np.fill_diagonal(D, 0.0)
         
            D[W == 0] = 0.0
    
        return A, W, D
    
    
    
    A_bin = (A_final > 0).astype(int)
    P_block_mat = loadmat('P_block.mat')
    P_block     = P_block_mat['P_block']
    labels     = P_block_mat['Labels']
    mu_block_W = P_block_mat['mu_block']
    sigma_block_W=P_block_mat['sigma_block']
    mu_block_D  =P_block_mat['D_mu_block']  
    sigma_block_D=P_block_mat['D_sigma_block']
    
    
    
    n_regs = n=20
    
    muW_intra,  muW_inter=215,5
    sigW_intra, sigW_inter=320,40
    # =========================
    muD_intra,  muD_inter =0.010,0.015
    
    sigD_intra, sigD_inter=0.005,0.005
    
    
    
    #%%
    
    
    
    
    
    _, W_use, D_use = generate_connectome_with_given_adjacency(
        A=A_bin,
        labels=labels,
        mu_block=mu_block_W,
        sigma_block=sigma_block_W,
        D_mu_block=mu_block_D,
        D_sigma_block=sigma_block_D,
        D_min=0.001,
        D_max=0.031,
        W_min=0.1,
        W_max=3500.0,
        seed=2028,
        weight_dist="gamma",
        delay_dist="normal_trunc"
    )
    
    
    #%%
                       
                                 
                           
    
    # A_bin = (A_final > 0).astype(int)
    
    # _, W_use, D_use = generate_connectome_with_given_adjacency(
    #     A=A_bin,
    #     labels=labels,
    #     mu_block=mu_block_W,
    #     sigma_block=sigma_block_W,
    #     D_mu_block=mu_block_D,
    #     D_sigma_block=sigma_block_D,
    #     D_min=D_min,
    #     D_max=D_max,
    #     W_min=0,
    #     W_max=3500,
    #     seed=2028,
    #     weight_dist="gamma",
    #     delay_dist="normal_trunc"
    # )
    
    np.fill_diagonal(W_use, 0.0)
    np.fill_diagonal(D_use, 0.0)
    D_use[W_use == 0] = 0.0
    
    plt.figure(figsize=(5, 5))
    plt.imshow(W_use, cmap="viridis")
    plt.title("Matriz sintética final W (pesos)")
    plt.colorbar(shrink=0.6, label="Peso")
    plt.tight_layout()
    
    plt.figure(figsize=(5, 5))
    plt.imshow(D_use, cmap="viridis")
    plt.title("Matriz sintética final D (delays)")
    plt.colorbar(shrink=0.6, label="Delay")
    plt.tight_layout()
    plt.show()
    
    #%%%
    
    import numpy as np
    import scipy.io as sio
    
    
    
    
    # sio.savemat("matrices_syn_seed_1002_con ajuste.mat", {
    #     "W_syn": W_use,
    #     "D_syn": D_use
    # })
    
    #%%
    
    # A_bin = (W_use > 0).astype(int)
    plt.figure(figsize=(5, 5))
    plt.imshow(A_syn, cmap="summer", interpolation="nearest")
    
    plt.title("Matriz adyacencia seed 1002")
    plt.xlabel("Columnas")
    plt.ylabel("Filas")
    plt.show()