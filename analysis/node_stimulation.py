#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 09:23:06 2023

@author: poloti
"""
import sys
sys.path.append('/home/poloti/Escritorio/KuramotoNetworksPackage-main/plot')
import numpy as np 
from networks import plotAAL90Brain as brain
import matplotlib.colors
custom_map = matplotlib.colors.LinearSegmentedColormap.from_list("custom",["silver","gold","orange","red"])
nodes = [87]
for i in nodes:
    node = np.zeros(90)
    node[i] = 1
    brain_name = 'node_stimulation %s'%(i)
    brain(node,orientation=[360,180],cmap_name=custom_map,namefile= brain_name +'1.jpeg')
    brain(node,orientation=[0,90],cmap_name=custom_map,namefile=brain_name +'2.jpeg')
    brain(node,orientation=[90,90],cmap_name=custom_map,namefile=brain_name +'3.jpeg')
    brain(node,orientation=[0,270],cmap_name=custom_map,namefile=brain_name +'4.jpeg')
    brain(node,orientation=[360,0],cmap_name=custom_map,namefile=brain_name +'5.jpeg')
