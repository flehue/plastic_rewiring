#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 08:42:25 2023

@author: poloti
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 11:20:50 2023

@author: poloti
"""
import sys 
sys.path.append('/home/poloti/Escritorio/KuramotoNetworksPackage-main/plot')
from scipy import signal
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np 
import os 
import seaborn as sns
import pandas as pd 
import matplotlib.colors
from networks import plotAAL90Brain as brain
directory='/home/poloti/Escritorio/simulaciones nuevas /'
direct = os.listdir(directory)
file_labels=sio.loadmat('/home/poloti/Escritorio/KuramotoNetworksPackage-main/input_data/AAL_labels')
#custom_map = matplotlib.colors.LinearSegmentedColormap.from_list("custom",["lightgray","cyan","lime","yellow","orange","red"])
#custom_map = matplotlib.colors.LinearSegmentedColormap.from_list("custom",["lightgray","yellow","orange","red","lime",])
custom_map = matplotlib.colors.LinearSegmentedColormap.from_list("custom",["lightgray","red","blue",])
#jet_map = matplotlib.colors.LinearSegmentedColormap.from_list("custom",["blue","cyan","lime","yellow","orange","red"])
jet_map = matplotlib.colors.LinearSegmentedColormap.from_list("custom",["blue","red","lime"])
cmap_after_before = matplotlib.colors.LinearSegmentedColormap.from_list("custom",["cyan","yellow","orange","red"])
labels=file_labels['label90']
nodes_inters_global = []
fff = []
for ss,element in enumerate (direct):
    if  element != 'config' and element != 'figuras':
        node_dir = directory +'/'+ element
        file_dict=sio.loadmat(node_dir)
        theta=file_dict['theta']
        node = eval(file_dict['nodes'][0])
        if type(node) != list:
            node = [node]
        before_stimul = theta[25000:300000,:].T
        stimul_signal =theta[300000:600000,:].T
        fstim = file_dict['fstim'][0]
        fff.append(fstim[0])
        f_index = {10.0:(40,48),13.0:(52,60),15.0:(60,68),23.0:(92,100),29.4:(118,126),40.0:(160,168),41.2:(166,174),43.0:(174,182)}
        a,b = f_index[fstim[0]]
        f, Pxx_den = signal.welch(np.sin(stimul_signal), 1000, nperseg=4096)
        f1, Pxx_den1 = signal.welch(np.sin(before_stimul), 1000, nperseg=4096)
        mean_after = np.mean(Pxx_den,axis=0)
        mean_bef = np.mean(Pxx_den1,axis=0)
        fig, (ax1, ax2) = plt.subplots(1, 2)
        title = 'General spectrum before-after stimulus fstim %s nodes %s'%(fstim[0],np.subtract(node,-1))
        title2 = 'Spectrum before stimulus fstim %s nodes %s'%(fstim[0],np.subtract(node,-1))
        title3 = 'Spectrum after stimulus fstim %s nodes %s'%(fstim[0],np.subtract(node,-1))
        title4 = 'Nodes Propagation for fstim %s nodes %s'%(fstim[0],np.subtract(node,-1))
        title5 = 'Spectrum nodes %s before-after stimulus fstim %s '%(fstim[0],np.subtract(node,-1))
        fig.suptitle(title)
        fig2,ax3 = plt.subplots()
        fig2.suptitle(title2)
        fig3,ax4= plt.subplots()
        fig3.suptitle(title3)
        fig4,ax5 = plt.subplots(figsize=(25,15))
        fig4.suptitle(title4)
        if len(node)==2:
            fig5,(ax6,ax7)= plt.subplots(1,2)
            fig5.suptitle(title5)
        else:
            fig5,(ax6,ax7)= plt.subplots(1,2)
            fig5.suptitle(title5)
        psd_cm_before = pd.DataFrame(Pxx_den1[:,a:b],index = np.arange(90),columns=np.round(f1[a:b],decimals=2))
        psd_cm_after = pd.DataFrame(Pxx_den[:,a:b],index = np.arange(90),columns=np.round(f[a:b],decimals=2))
        sns.heatmap(psd_cm_before,cmap=cmap_after_before,ax=ax3)
        sns.heatmap(psd_cm_after,cmap=cmap_after_before,ax=ax4)
        nodes_inters = []
        nodes_inters_cmap = []
        count=0
        print(node)
        for i in range (len(Pxx_den1)):
            if len(node) == 2:
                if i != node[0] and i != node[1]:        
                    ax1.plot(f1,Pxx_den1[i],'blue')
                    ax2.plot(f,Pxx_den[i],'blue')
                else:
                    ax6.plot(f1,Pxx_den1[i],label=node[count])
                    ax7.plot(f,Pxx_den[i],label=node[count])
                    count += 1 
            elif len(node) == 1:
                if i != node[0]:        
                    ax1.plot(f1,Pxx_den1[i],'blue')
                    ax2.plot(f,Pxx_den[i],'blue')
                else:
                    ax6.plot(f1,Pxx_den1[i],label=node[count])
                    ax7.plot(f,Pxx_den[i],label=node[count])
            node_inters = np.mean(Pxx_den[i,a:b])/np.mean(Pxx_den1[i,a:b])
            if node_inters >= 2:
                nodes_inters_cmap.append([node_inters])
                #nodes_inters.append(node_inters)
                nodes_inters.append(0.3)
            else:
                nodes_inters_cmap.append([0])
                nodes_inters.append(0)
        #print(nodes_inters)
     
        print(nodes_inters)
        ax1.plot(f1,mean_bef,'red',label = 'mean')
        ax1.set_xlim(0,60)
        ax2.plot(f,mean_after,'red',label = 'mean')
        ax2.set_xlim(0,60)
        ax6.set_xlim(0,60)
        ax7.set_xlim(0,60)
        if len(node) == 2:
            nodes_inters_cmap[node[0]] = [1.0]
            nodes_inters_cmap[node[1]] = [1.0]
            #nodes_inters[node[0]] = np.max(nodes_inters) + 5
            #nodes_inters[node[1]] = np.max(nodes_inters) + 5
            nodes_inters[node[0]] = 1.0
            nodes_inters[node[1]] = 1.0
        else:
            nodes_inters_cmap[node[0]] = [1.0]
            #nodes_inters[node[0]] = np.max(nodes_inters) + 5
            nodes_inters[node[0]] = 1.0
        nodes_inters = np.array(nodes_inters)
        if np.max(nodes_inters) != 0 and np.max(nodes_inters_cmap) !=0:
            nodes_inters = nodes_inters/np.max(nodes_inters)
            nodes_inters_cmap= nodes_inters_cmap/np.max(nodes_inters_cmap)
        else:
            None
        #nodes_inters_cmap = np.array(nodes_inters_cmap)
        
        nodes_inters_global.append(nodes_inters)
        brain_name = directory +'/'+'figuras/'+ "Node %s propagation for fstim %s"%(np.subtract(node,-1),fstim)
        #print(nodes_inters)
        brain(nodes_inters,nodes_stimul=node,orientation=[360,180],cmap_name=custom_map,namefile= brain_name +'1.jpeg')
        brain(nodes_inters,nodes_stimul=node,orientation=[0,90],cmap_name=custom_map,namefile=brain_name +'2.jpeg')
        brain(nodes_inters,nodes_stimul=node,orientation=[90,90],cmap_name=custom_map,namefile=brain_name +'3.jpeg')
        brain(nodes_inters,nodes_stimul=node,orientation=[0,270],cmap_name=custom_map,namefile=brain_name +'4.jpeg')
        brain(nodes_inters,nodes_stimul=node,orientation=[360,0],cmap_name=custom_map,namefile=brain_name +'5.jpeg')
        #nodes_inters_global.append(nodes_inters)
        #fig.savefig(title+'.jpeg',dpi=400)
        #fig2.savefig(title2+'.jpeg',dpi=400)
        #fig3.savefig(title3+'.jpeg',dpi=400)
nodes_inters_global = np.array(nodes_inters_global)
propagatio = pd.DataFrame(nodes_inters_global,index=fff,columns=labels)
sns.heatmap(propagatio,linewidths = 0.75,linecolor = "black",cmap=jet_map,ax=ax5,square=True,xticklabels=labels,cbar=False)
fig4.savefig(directory +'/'+'figuras/'+title4+'.jpeg',dpi=400)
        #fig5.savefig(directory +'/'+title5+'.jpeg',dpi=400)