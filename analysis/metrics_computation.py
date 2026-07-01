#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 06:52:03 2025

@author: bdl
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import frequency

N=140
MD_all_values = np.arange(0, 26, 1)  # de 0 a 30 en pasos de 1

# MD_values=MD_values*1e-3

K_all_values = np.arange(0, 5, 0.5)  # de 100 a 1100 en pasos de 50


for K_val in K_all_values:
  
    for MD_val in MD_all_values:
     
            
            k_adjusted = K_val * N
            md_adjusted = MD_val * 1e-3
           
            file_path = f'F:/Monica_paper/20_03/K{K_val}MD{MD_val}/{freq:.1f}_400.0_[{node-1}]_nodeSimulations_weight_N90_K{k_adjusted:.3f}_MD{md_adjusted:.3f}_seed789.mat'
            
           
            file_data = sio.loadmat(file_path)
            theta = file_data['theta']
            
            f, Pxx, pfreqs = frequency.peak_freqs(theta.T, fs=1000, nperseg=5000, noverlap=2500, applySin=True, includeDC=False)
            Hxx_new = frequency.spectralEntropy(Pxx)
            Hxx_sum = np.sum(Hxx_new)
                
                
kop_mean=file['KOP']
normKOP=plt.Normalize(vmin=0,vmax=1)
kop_std=file['SD_KOP']
normKOPSD=plt.Normalize(vmin=0,vmax=np.max(kop_std)//0.1*0.1)
Pxx=file['Pxx']
H=file['H']
# mean_pf=file['mean_pf']
normH=plt.Normalize(vmin=np.min(H)//5*5,vmax=np.max(H)//5*5+5)
max_index=np.argmax(H)
max_K=max_index//41
max_MD=max_index-max_K*41

# K_all_values=np.arange(0,10.5,0.5)
# MD_all_values=np.arange(0,41,1)
freqs=np.linspace(0,80,401)

peak_Pxx=freqs[np.argmax(Pxx,axis=2)]

fig1=plt.figure(figsize=(4.5,3.5))
ax1=fig1.add_subplot(1,1,1)
im1=ax1.imshow(np.flipud(H),cmap=plt.cm.turbo,aspect='equal',interpolation='None',norm=normH)
ax1.plot(max_MD,20-max_K,'s',color='white',mfc='none',markersize=3.85)
ax1.plot(40,0,'s',color='cyan',mfc='none',markersize=3.85)
ax1.plot(38,20-18,'s',color='lightgrey',mfc='none',markersize=3.85)
ax1.plot(20,20-10,'s',color='indigo',mfc='none',markersize=3.85)
ax1.plot(16,20-6,'s',color='magenta',mfc='none',markersize=3.85)
cb1=fig1.colorbar(im1,ax=ax1,shrink=0.5)
ax1.set_xlabel('mean delay [MD] (ms)',fontsize=8,labelpad=0)
ax1.set_ylabel('global coupling [K]',fontsize=8)
ax1.set_yticks(np.arange(0,len(K_all_values),2))
ax1.set_yticklabels(np.flip(np.arange(0,len(K_all_values),2))*0.5)
ax1.set_xticks(np.arange(0,len(MD_all_values),5))
ax1.set_xticklabels(np.arange(0,len(MD_all_values),5))
ax1.tick_params('both',labelsize=8)
cb1.set_label('spectral entropy [H] (nits)',fontsize=8)
cb1.set_ticks([10,20,30,40,50,60])
cb1.set_ticklabels([10,20,30,40,50,60],fontsize=8)

# fig1.savefig('Fig1.tif',dpi=600,pil_kwargs={"compression": "tiff_lzw"},bbox_inches='tight')

fig1_spectrums=plt.figure(figsize=(4.5,3.5))
gs=gridspec.GridSpec(len(K_all_values), len(MD_all_values))
for k,K in enumerate(K_all_values):
    for md, MD in enumerate(MD_all_values):
        ax=fig1_spectrums.add_subplot(gs[len(K_all_values)-k-1,md])
        ax.plot(freqs,Pxx[k,md],'k',linewidth=0.5)
        ax.set_axis_off()
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_yticklabels([])
# fig1_spectrums.savefig('Fig1_spectrum.png',dpi=600,bbox_inches='tight')

figS1=plt.figure(figsize=(4.5,6))
g1s=gridspec.GridSpec(2, 1)
axA=figS1.add_subplot(g1s[0])
imA=axA.imshow(np.flipud(kop_mean),cmap=plt.cm.jet,aspect='equal',interpolation='None',norm=normKOP)
cbA=figS1.colorbar(imA,ax=axA,shrink=0.9)
axA.set_xlabel('mean delay [MD] (ms)',fontsize=8,labelpad=0)
axA.set_ylabel('global coupling [K]',fontsize=8)
axA.set_yticks(np.arange(0,len(K_all_values),2))
axA.set_yticklabels(np.flip(np.arange(0,len(K_all_values),2))*0.5)
axA.set_xticks(np.arange(0,len(MD_all_values),5))
axA.set_xticklabels(np.arange(0,len(MD_all_values),5))
axA.tick_params('both',labelsize=8)
axA.plot(max_MD,20-max_K,'s',color='gray',mfc='none')
axA.text(-0.15,1,'A',transform=axA.transAxes)
cbA.set_label(r'$\langle$ KOP $\rangle$',fontsize=8)


axB=figS1.add_subplot(g1s[1])
imB=axB.imshow(np.flipud(kop_std),cmap=plt.cm.jet,aspect='equal',interpolation='None',norm=normKOPSD)
cbB=figS1.colorbar(imB,ax=axB,shrink=0.9)
axB.set_xlabel('mean delay [MD] (ms)',fontsize=8,labelpad=0)
axB.set_ylabel('global coupling [K]',fontsize=8)
axB.set_yticks(np.arange(0,len(K_all_values),2))
axB.set_yticklabels(np.flip(np.arange(0,len(K_all_values),2))*0.5)
axB.set_xticks(np.arange(0,len(MD_all_values),5))
axB.set_xticklabels(np.arange(0,len(MD_all_values),5))
axB.tick_params('both',labelsize=8)
axB.plot(max_MD,20-max_K,'s',color='gray',mfc='none')
axB.text(-0.15,1,'B',transform=axB.transAxes)
cbB.set_label('sd(KOP)',fontsize=8)


# axC=figS1.add_subplot(g1s[2])
# #Average of the peaks of each node
# imC=axC.imshow(np.flipud(mean_pf),cmap=plt.cm.jet,aspect='equal',interpolation='None',vmin=0,vmax=50)
# #Peak of the average spectrum
# # imC=axC.imshow(np.flipud(peak_Pxx),cmap=plt.cm.jet,aspect='equal',interpolation='None',vmin=0,vmax=40)
# cbC=figS1.colorbar(imC,ax=axC,shrink=0.9)
# axC.set_xlabel('mean delay [MD] (ms)',fontsize=8,labelpad=0)
# axC.set_ylabel('global coupling [K]',fontsize=8)
# axC.set_yticks(np.arange(0,len(K_all_values),2))
# axC.set_yticklabels(np.flip(np.arange(0,len(K_all_values),2))*0.5)
# axC.set_xticks(np.arange(0,len(MD_all_values),5))
# axC.set_xticklabels(np.arange(0,len(MD_all_values),5))
# axC.tick_params('both',labelsize=8)
# axC.plot(max_MD,20-max_K,'s',color='gray',mfc='none')
# axC.text(-0.15,1,'C',transform=axC.transAxes)
# cbC.set_label('Hz',fontsize=8)

figS1.tight_layout()
# figS1.savefig('FigS1.tif',dpi=600,pil_kwargs={"compression": "tiff_lzw"},bbox_inches='tight')