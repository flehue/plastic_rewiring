#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 09:53:08 2023

@author: felipe
"""
import frequency
import synchronization 
import numpy as np
import sys
import os
sys.path.append(os.path.abspath('../../'))
# import analysis.frequency as frequency
# import analysis.synchronization as synchronization
import scipy.io as sio
import scipy.signal as signal
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import gc

nperseg=5000 
noverlap=2500
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
directory='/home/poloti/Escritorio/simulaciones nuevas /29.4_400.0_[87]_nodeSimulations_weight_N90_K360.000_MD0.021_seed789'

#directory='/media/felipe/Elements/Kuramoto_HeatMap/'
#Different K
K=4
#Different MD
MD=0.021
#Different seed
for n, seed in enumerate([789]):
    print('Seed:',seed)
    #filename=directory+'MaximumSE_N%d_K%.3F_MD%.3f_seed%d.mat'%(N,K*90,MD,seed)
    file_dict=sio.loadmat(directory)
    theta=file_dict['theta']
    theta=theta[600000::,:]
    f=np.linspace(0,fs/2,nperseg//2+1)
    # f,t,Sxx_nodes=frequency.spectrogram(np.sin(theta.T),nperseg=nperseg,noverlap=noverlap)
    # Pxx_mean=np.mean(np.sum(Sxx_nodes,axis=-1),axis=0)
    # peaks_indexes_mean=frequency.findPeaksScipy(Pxx_mean,tolPercentile=1,tolF=5,power_quotient=0.01)
    freq_peaks=f[[65,75,147,206,215]] #13,15,29.4,41.2,43
    
        
    tt=np.arange(0,1772,dt)
    
    fbands=[]
    f_lows=np.zeros_like(freq_peaks)
    f_highs=np.zeros_like(freq_peaks)
    for fp,freq in enumerate(freq_peaks):
        f_lows[fp]=freq-0.5
        f_highs[fp]=freq+0.5
        fbands.append('%.2f-%.2f Hz'%(f_lows[fp],f_highs[fp]))
    ###Needed for figures#####
    # maxSpectrum=10*np.log10(np.max(np.sum(Sxx_nodes[:,0:80,:],axis=2)))
    # b30,a30=signal.butter(4,2*30/fs,btype='lowpass')
    # filtered30=signal.filtfilt(b30,a30,np.sin(theta.T))
    
    for threshold in [0.232]:#0.5,0.9
        for fb,fband in enumerate(fbands):
            print(threshold,fband)
            
            
            envelopes_low=synchronization.envelopesFrequencyBand(theta.T,f_low=f_lows[fb],f_high=f_highs[fb],fs=1000,applyLow=False)
            th_envelopes=np.ones_like(envelopes_low)*threshold
            binary_envelopes=np.zeros_like(envelopes_low)
            
            for n in range(N):
                binary_envelopes[n,:]=envelopes_low[n,:]>th_envelopes[n]
            
            ##FC from the envelopes are independent of the threshold

            #square2=binary_envelopes.dot(binary_envelopes.T)/len(tt)
            #FC=np.corrcoef(envelopes_low)
            #FCD,corr_vectors,shift=synchronization.extract_FCD(binary_envelopes,wwidth=nperseg,maxNwindows=600,olap=0.5)
            #np.savez('FCcorr_fb%s_K=%d_MD=%d_seed%d.npz'%(fband,K,MD*1000,seed),FC=FC,FCD=FCD,corr_vectors=corr_vectors,shift=shift)
            #del corr_vectors,FC,FCD,shift, square2 #don't delete if figures are required
                
            #Co-occurrences
            durations, occupancy, co_occurrences=synchronization.extractTimeStatisticsEvents(binary_envelopes,min_duration=5)
            np.savez('./result/cooccurrences_fixed_%.3f_fb%s_K=%d_MD=%d_seed%d.npz'%(threshold,fband,K,MD*1000,seed),occupancy=occupancy,durations=durations,co_occurrences=np.array(co_occurrences,dtype=object))
            ########Figures#####################
            # fig=plt.figure(figsize=(8,9))
            # gs=gridspec.GridSpec(3,4,width_ratios=[1,1,1,1], height_ratios=[1,1,0.6],wspace=0.6,hspace=0.3)
            # axA=fig.add_subplot(gs[0,:])
            # axC=fig.add_subplot(gs[1,:])
            # axB=fig.add_subplot(gs[2,0])
            # axD=fig.add_subplot(gs[2,1])
            # axE=fig.add_subplot(gs[2,2])
            # axF=fig.add_subplot(gs[2,3]) 
            # binary_envelopes_mask=np.ma.masked_where(binary_envelopes<1, binary_envelopes)
            # for n,nn in enumerate(np.arange(0,90,3)):
            #     axA.plot(tt[50000:75000],filtered30[nn,54000:79000]+2.5*n,color=colors[n],linewidth=0.2)
            #     axA.fill_between(tt[50000:75000], y1=np.zeros_like(tt[50000:75000])+2.5*n,y2=np.zeros_like(tt[50000:75000])+2.5*n+1.5, where=binary_envelopes_mask[nn,50000:75000], alpha=0.2,color=colors[n])
            #     axC.plot(tt,envelopes_low[nn,0:len(tt)]+2.5*n,color=colors[n],linewidth=0.2)
            #     axC.fill_between(tt, y1=np.zeros_like(tt)+2.5*n,y2=np.zeros_like(tt)+2.5*n+1.5, where=binary_envelopes_mask[nn,0:len(tt)], alpha=0.2,color=colors[n])
            #     axB.plot(f[0:320],10*np.log10(np.sum(Sxx_nodes[nn,0:320,:],axis=1))/maxSpectrum+2*n,color=colors[n])
            # axB.fill_betweenx(y=np.arange(60),x1=np.ones((60,))*f_lows[fb],x2=np.ones((60,))*f_highs[fb],color=plt.cm.tab10(7),alpha=0.2)
           
            # imD=axD.imshow(FC,aspect='auto',cmap=plt.cm.RdBu_r,vmin=-1,vmax=1)
            # imF=axF.imshow(square2,aspect='auto',cmap=plt.cm.turbo,vmin=0,vmax=1)
            
            # axB.set_xticks([0,20,40,60,80])
            # axB.set_xticklabels([0,20,40,60,80])
            # axA.set_yticklabels('')
            # axC.set_yticklabels('')
            # axB.set_yticklabels('')
            # plt.colorbar(imD,ax=axD)
            # plt.colorbar(imF,ax=axF)
            # axA.set_ylabel('sin(theta) (below 30Hz)')
            # axC.set_ylabel('envelope %s'%fband)
            # axE.set_ylabel('Sum binary env.')
            # axA.set_xlabel('time (s)')
            # axC.set_xlabel('time (s)')
            # axE.set_xlabel('time (s)')
            # axE.plot(tt,np.sum(binary_envelopes,axis=0)[0:len(tt)])
            # axE.set_ylim([-1,91])
            # axB.set_xlabel('frequency (Hz)')
            # axB.set_ylabel('Spectrum')
            # axD.set_xlabel('Nodes')
            # axD.set_ylabel('FC env.')
            # axF.set_xlabel('Nodes')
            # axF.set_ylabel('Co-occurrence')
            # fig.suptitle('K=%d, MD=%d Freq:%s'%(K,MD*1000,fband),y=0.93)
            # fig.savefig('AAL90_fixed_%.2f_fb%s_K=%d_MD=%d_seed%d.png'%(threshold,fband,K,MD*1000,seed),dpi=300)
            # plt.close()
            
            
            # fig2=plt.figure(figsize=(8,9))
            # gs2=gridspec.GridSpec(3,1, height_ratios=[1,1,1],wspace=0.3,hspace=0.3)
            # axA2=fig2.add_subplot(gs2[0])
            # axB2=fig2.add_subplot(gs2[1])
            # axC2=fig2.add_subplot(gs2[2])
            
            # axA2.hist(envelopes_low[np.argmax(np.sum(envelopes_low,axis=1)),:],bins=np.arange(0,1.05,0.05))
            # axB2.hist(durations)
            # axC2.plot(occupancy,'o')
         
            # fig2.savefig('AAL90_results_%.2f_fb%s_K=%d_MD=%d_seed%d.png'%(threshold,fband,K,MD*1000,seed),dpi=300)
            # del binary_envelopes_mask
            # plt.close()
           
           
            
            
            
            del envelopes_low, th_envelopes, binary_envelopes, durations, co_occurrences,

            gc.collect()
        gc.collect()
    del theta
    gc.collect()
        






