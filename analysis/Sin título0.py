#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 01:52:14 2025

@author: bdl
"""

# Python standard libraries
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.io as sio
import re
from scipy.stats import zscore
import pandas as pd
# KuramotoNetworksPackage library
import sys
import os
sys.path.append(os.path.abspath( '../' ))

# Plotting networks in brain surfaces
# import plot . networks as pltn

# Calculate structural features
import analysis . connectivityMatrices as connectivityMatrices
# Calculate node - dynamical features
import analysis . frequency as frequency

# Calculate network - dynamical features
import analysis . synchronization as synchrony

#visualization
import seaborn as sns


#%%
# Time - frequency parameters
fs = 1000
nperseg=5000
noverlap =nperseg/2

#%%  MULTIPROCESSING


def multiprocessing(N, directory,result_path):
        
    simulation_files=[]
    for file_name in os.listdir(directory):
        if file_name.split('.')[3]=='mat':
            simulation_files.append(file_name)
            
    #ordenar los files
    # def myFunc(e):
    #     return float(e.split('_')[2].replace('k', '')) + float(e.split('_')[3].replace('MD', ''))
    def myFunc(e):
        parts = e.split('_')
        k_val = float(parts[1].replace('K', ''))      # "K0.000" → 0.000
        md_val = float(parts[2].replace('MD', ''))    # "MD0.040" → 0.040
        return k_val + md_val
    simulation_files.sort(key=myFunc)    
    
    #declarar parametros
    
    kop_matrix=np.zeros((12,41))
    kop_lista=[]
    
    sd_kop_matrix=np.zeros((12,41))
    sd_kop_lista=[]
    
    spectral_entropy_matrix=np.zeros((12,41))
    spectral_entropy_lista=[]
    
    average_fpeaks_matrix=np.zeros((12,41))
    average_fpeaks_lista=[]
    
    for filename in simulation_files:
        file=sio.loadmat(directory + filename)
        print(filename)
        #average KOP
        kop_lista.append(np.mean(file ['kop'][0]))
        # SD kop
        sd_kop = np.std(file ['kop'][0] )
        sd_kop_lista.append(sd_kop)
        # Welch PSD
        f , Pxx , fpeaks = frequency.peak_freqs ( file['theta'][20000: ,:].T , fs = fs , nperseg = nperseg ,noverlap = noverlap , applySin = True )
        #average fpeaks
        average_fpeaks_lista.append(np.mean(fpeaks))
        # Spectral Entropy
        H_nodes = frequency.spectralEntropy ( Pxx )
        globalH = np.sum(H_nodes) #/len(H_nodes)
        spectral_entropy_lista.append(globalH)
        
        
         
    indice=0
    for k in range(12):
        for md in range(41):
            kop_matrix[k,md]=kop_lista[indice]
            sd_kop_matrix[k,md]=sd_kop_lista[indice]
            average_fpeaks_matrix[k,md]=average_fpeaks_lista[indice]
            spectral_entropy_matrix[k,md]=spectral_entropy_lista[indice]
            indice+=1
    
    
    #%% imagen SD_KOP
    
    # result_path=f"graficas parametros/matrices_reagrupadas_v6/{N}ROI_subc4Hz/"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    # Graficar la imagen con un mapa de colores
    
    ylabel=list(np.arange(0, 6, 0.5))
    
    # ylabel=list(np.arange(1000,-50,-50))
    
    plt.figure()
    ax=sns.heatmap(sd_kop_matrix,yticklabels=ylabel, xticklabels='auto',cmap='jet')
    plt.xlabel('mean delay (MD)')
    plt.ylabel('global coupling (K)')
    plt.xticks(rotation=90)
    plt.title('SD_KOP')
    plt.tight_layout()
    fig = plt.gcf()
    fig.savefig(result_path + 'SD_KOP.jpg', dpi=600, bbox_inches='tight')
    # plt.close(fig)
    
    np.save(result_path + 'SD_KOP.npy', sd_kop_matrix)
    
    #otra opcion mas viejita
    # plt.figure()
    # plt.imshow(sd_kop_matrix, cmap='jet')  # Puedes cambiar 'viridis' por otro colormap de tu elección
    # plt.colorbar()  # Mostrar la barra de colores (escala)
    # # plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
    # plt.show()
    
    #%% imagen spectral_entropy
    
    plt.figure()
    ax=sns.heatmap(spectral_entropy_matrix ,yticklabels=ylabel, xticklabels='auto',cmap='jet')
    plt.xlabel('mean delay (MD)')
    plt.ylabel('global coupling (K)')
    ax.invert_yaxis() 
    plt.xticks(rotation=90)
    plt.title('Spectral Entropy')
    plt.tight_layout()
    fig = plt.gcf()
    fig.savefig(result_path + 'Spectral_Entropy.jpg', dpi=600, bbox_inches='tight')
    # plt.close(fig)
    
    np.save(result_path + 'Spectral_Entropy.npy', spectral_entropy_matrix)
    
    #tamaño(cantidad de pares de parametros) de la region SE>0.5        justificar por que 0.5 umbral
    average_SE=np.mean(spectral_entropy_matrix)
    max_SE=np.max(spectral_entropy_matrix)
    area_mayorSE=len(np.where(spectral_entropy_matrix>0.5)[0])
    
    ady_SE_matrix=np.where(spectral_entropy_matrix>=0.5,1,0)
    
    plt.figure()
    ax=sns.heatmap(ady_SE_matrix ,yticklabels=ylabel, xticklabels='auto',cmap='jet')
    plt.xlabel('mean delay (MD)')
    plt.ylabel('global coupling (K)')
    ax.invert_yaxis() 
    plt.xticks(rotation=90)
    plt.title('Spectral Entropy_binaria area mayor SE')
    plt.tight_layout()
    fig = plt.gcf()
    fig.savefig(result_path + 'Spectral_Entropy_binaria.jpg', dpi=600, bbox_inches='tight')
    # plt.close(fig)
    
    np.save(result_path + 'SE_area_binaria.npy', ady_SE_matrix)

    
    #%%  imagen average KOP
    
    # Graficar la imagen con un mapa de colores
    plt.figure()
    ax=sns.heatmap(kop_matrix ,yticklabels=ylabel, xticklabels='auto',cmap='jet')
    plt.xlabel('mean delay (MD)')
    plt.ylabel('global coupling (K)')
    ax.invert_yaxis() 
    plt.xticks(rotation=90)
    plt.title('Average KOP')
    plt.tight_layout()
    fig = plt.gcf()
    fig.savefig(result_path + 'Average_KOP.jpg', dpi=600, bbox_inches='tight')
    # plt.close(fig)
    
    np.save(result_path + 'Average_KOP.npy', kop_matrix)

    
    #%%  imagen average fpeaks
    
    # Graficar la imagen con un mapa de colores
    # plt.figure()
    # ax=sns.heatmap(average_fpeaks_matrix ,yticklabels=ylabel, xticklabels='auto',cmap='jet')
    # plt.xlabel('mean delay (MD)')
    # plt.ylabel('global coupling (K)')
    # plt.xticks(rotation=90)
    # plt.title('Peak frecuency (Hz)')
    # plt.tight_layout()
    # fig = plt.gcf()
    # fig.savefig(result_path + 'Peak_frecuency_(Hz).jpg', dpi=600, bbox_inches='tight')
    # # plt.close(fig)
    # # 
    # np.save(result_path + 'Peak_frecuency.npy', average_fpeaks_matrix)

    
    #%%  imagen espectro
    # plt.figure()
    # plt.plot(f , np . mean ( Pxx , axis =0) , linewidth =2 , color = plt . cm . tab20 (0) ,
    # label = 'Average PSD')
    # plt . plot (f , Pxx .T , ':', color = plt . cm . tab20 (1) , label = 'Individual PSD')
    # plt . xlabel ('frequency ( Hz )')
    # plt . ylabel ( r' PSD $ ( u ^2/ Hz ) $ ')
    # plt.title('Average PSD')
    # plt.xlim([0,80])
    # plt.tight_layout()
    # fig = plt.gcf()
    # fig.savefig(result_path + 'Average_PSD.jpg')
    # # plt.close(fig)
    
    
    #%%
    # k y MD de mayor spectral entropy
    max_se=np.max(spectral_entropy_matrix)
    print('maximal spectral entropy', max_se)
    K_se,MD_se=np.where(spectral_entropy_matrix==np.max(spectral_entropy_matrix))
    K_se=((21-K_se-1)/2)[0]*100
    print('valores de k y MD para la maxima spectral entropy', K_se, MD_se[0])
    
    
    
    # k y MD de mayor sd_kop
    max_sdkop=np.max(sd_kop_matrix)
    print('maxima sd_kop', max_sdkop)
    K_sd,MD_sd=np.where(sd_kop_matrix==np.max(sd_kop_matrix))
    K_sd=((21-K_sd-1)/2)[0]*100
    print('k y MD para el maximo SD KOP', K_sd,MD_sd[0])
    
    return(K_se, MD_se[0], K_sd, MD_sd[0], area_mayorSE, ady_SE_matrix)


#%%entradas

levels=[90]
# directory=f'/media/yarelis/Elements/Yarelis/simulaciones Kuramoto/matrices_AAL90/{N}ROI/'
# directory=f'/media/yarelis/Elements/Yarelis/simulaciones Kuramoto/matrices reagrupadas v6/{N}ROI_subc4Hz/'



ady_SE_matrix={}
area_mayorSE={}
K_se={}
MD_se={}
K_sd={}
MD_sd={}


for N in levels:
    # directory=f'/media/yarelis/Elements/Yarelis/simulaciones Kuramoto/matrices reagrupadas v6/{N}ROI_subc4Hz/'
    directory='/home/bdl/Descargas/exploracion/KuramotoNetworksPackage-main/KuramotoNetworksPackage-main/output_timeseries/N140_K_12_08_40hz_only/'
    # result_path=f"graficas parametros/matrices_reagrupadas_v6/{N}ROI_subc4Hz/"
    result_path=f"graficas parametros/matrices_AAL90/"
    K_se[N], MD_se[N], K_sd[N], MD_sd[N], area_mayorSE[N], ady_SE_matrix[N]=multiprocessing(N,directory,result_path)


#%%





















