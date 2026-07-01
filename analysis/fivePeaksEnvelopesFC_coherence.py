#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 16:28:29 2023

@author: felipe
"""

import numpy as np
import sys
import os
sys.path.append(os.path.abspath('../../'))
import frequency
import synchronization
import scipy.io as sio
import scipy.signal as signal
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import gc
import tqdm
from multiprocessing import Lock
import concurrent.futures
import time

def RunFCD(seed):
    time.sleep(0.1)
    dt=1e-3
    fs=1000
    #t=np.arange(20,300,dt)
    N=90

    colors=[plt.cm.tab10(0),plt.cm.tab10(1),plt.cm.tab10(2),plt.cm.tab10(3),
            plt.cm.tab10(4),plt.cm.tab10(5),plt.cm.tab10(6),plt.cm.tab10(7),
            plt.cm.tab10(8),plt.cm.tab10(9),plt.cm.Dark2(0),plt.cm.Dark2(1),
            plt.cm.Dark2(2),plt.cm.Dark2(3),plt.cm.Dark2(4),plt.cm.Dark2(5),
            plt.cm.Dark2(6),plt.cm.Dark2(7),plt.cm.Set1(0),plt.cm.Set1(1),
            plt.cm.Set1(2),plt.cm.Set1(3),plt.cm.Set1(4),plt.cm.Set1(5),
            plt.cm.Set1(6),plt.cm.Set1(7),plt.cm.Set1(8),plt.cm.Set2(0),
            plt.cm.Set2(1),plt.cm.Set2(2),plt.cm.Set2(3),plt.cm.Set2(4),
            plt.cm.Set2(5),plt.cm.Set2(6),plt.cm.Set2(7)]


    optimalK=4     
    directory='/home/poloti/Escritorio/Carpeta sin título/nodos times different/300/'
    #directory='/media/felipe/Elements/Kuramoto_HeatMap/'



    #Different K
    K=4
    #Different MD
    MD=0.021

    nperseg=5000
    noverlap=2500
    f=np.linspace(0,fs/2,nperseg//2+1)
    freq_peaks=f[[65,75,147,206,215]] #13,15,29.4,41.2,43
    freq_indexes=[65,75,147,206,215]
        
    # tt=np.arange(0,1780,dt)
    Twindow=9*fs
    fbands=[]
    f_lows=np.zeros_like(freq_peaks)
    f_highs=np.zeros_like(freq_peaks)
    for fp,freq in enumerate(freq_peaks):
        f_lows[fp]=freq-0.5
        f_highs[fp]=freq+0.5
        fbands.append('%.2f-%.2f Hz'%(f_lows[fp],f_highs[fp]))

    #pointsFC=[384,334,170,122,118] #5 cycles
    pointsFC=[231,200,102,73,70] #3 cycles
    
    print('Seed:',seed)
    filename=directory+'40.0_200.0_None_nodeSimulations_weight_N90_K360.000_MD0.021_seed%d'%(seed)
    file_dict=sio.loadmat(filename)
    file_theta=file_dict['theta'][20000::,:]
    for n in range(31):
        sinTheta=np.sin(file_theta[n*Twindow:(n+1)*Twindow,:]).T
        for fb1,fband1 in enumerate(tqdm.tqdm(range(1))):
            fband=fbands[4]
            fb=4
            # envelopes_low=synchronization.envelopesFrequencyBand(theta.T,f_low=f_lows[fb],f_high=f_highs[fb],fs=1000,applyLow=True)
            # FC=np.corrcoef(envelopes_low)
            FCD,corr_vectors,shift_amplitude=synchronization.extract_FCD(sinTheta,wwidth=pointsFC[fb]*3,wcoh=pointsFC[fb],maxNwindows=100000,olap=0.5,nfft=nperseg,freq_index=freq_indexes[fb],mode='ccoh')
            np.savez('/home/poloti/Escritorio/Carpeta sin título/nodos times different/result_300/FC/Amplitud/FCenvelopes_Coherence_Amplitude_fb%s_K=%d_MD=%d_seed%d_n%d.npz'%(fband,K,MD*1000,seed,n),FCD=np.abs(FCD),corr_vectors=np.abs(corr_vectors),shift=shift_amplitude)
            np.savez('/home/poloti/Escritorio/Carpeta sin título/nodos times different/result_300/FC/Phase/FCenvelopes_Coherence_Phase_fb%s_K=%d_MD=%d_seed%d_n%d.npz'%(fband,K,MD*1000,seed,n),FCD=np.imag(FCD),corr_vectors=np.imag(corr_vectors),shift=shift_amplitude)
            np.savez('/home/poloti/Escritorio/Carpeta sin título/nodos times different/result_300/FC/Real/FCenvelopes_Coherence_Real_fb%s_K=%d_MD=%d_seed%d_n%d.npz'%(fband,K,MD*1000,seed,n),FCD=np.real(FCD),corr_vectors=np.real(corr_vectors),shift=shift_amplitude)
            
            del corr_vectors,FCD,shift_amplitude #don't delete if figures are required
            gc.collect()
        del sinTheta
        gc.collect()
    del file_theta

#Different seed  #3,5
seeds=[11,14,34,41,48,69,566,789,999,1464]#[8,13,21,34,55,89,144,233]
for j in range(1):
    print('Starting FCD calculation')
    lock = Lock()
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        executor.map(RunFCD, seeds)
    
    
