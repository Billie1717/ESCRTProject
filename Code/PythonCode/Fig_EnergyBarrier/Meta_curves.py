#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 08:28:29 2021

@author: billiemeadowcroft
"""
import matplotlib.pyplot as plt
#import NumbMetrop2subs2
import Metrop2Bindings3subs
import numpy as np


#Dataloc = str('/Users/billiemeadowcroft/Documents/PHD/Code/Numba_ReactPath/Data/GifSweep/')
#Dataloc = str('/Users/billie/Documents/PHD/Curvy_model/GitPlots/MechanoChemDataAndPlotting/Fig_EnergyBarrier/Data/')
Dataloc = str('/Users/billiemeadowcroft/Documents/PHD/Curvy_model/GitPlots/MechanoChemDataAndPlotting/Fig_EnergyBarrier/Data/')
fname = str('bigspaceTest1_')


Curve0 = 8.0
Curve1 = 2.5
Curve2 = 1.0 #Curvatures
k_0= 0.8
k_1= 0.8
k_2 = 0.8
k_mem = 2.5 # bending rigidities
pitch0 = 2.0
pitch1 = 1.0
pitch2 = 0.0 # pitch/tilt
BindMem2 = -0.8
BindMem0 = BindMem2*11.5
BindMem1 = BindMem2*3.2


trials = int(100)
switches = int(300)
No_oth = int(10)

kwargs = dict(trials=trials,switches=switches,No_oth=No_oth,Curve0=Curve0,Curve1=Curve1,Curve2=Curve2,k_0=k_0,k_1=k_1,k_2=k_2,k_mem=k_mem,
              pitch0=pitch0,pitch1=pitch1,pitch2=pitch2,BindMem0=BindMem0,BindMem1=BindMem1,BindMem2=BindMem2)


params = Metrop2Bindings3subs.Params(**kwargs)
print(params)

y=Metrop2Bindings3subs.main_numba(params)[0]
zeroth = Metrop2Bindings3subs.main_numba(params)[1]
print(np.shape(y))
#print("should be zero", zeroth)
N = int(42)

for b in range(N):
    if b == 0:        
        f=open(Dataloc+fname +str(No_oth)+'3subs_'+str(trials)+'_'+str(switches)+'.txt','w')
        np.savetxt(f, y[b],  delimiter=',', header=str(params))
        f.close()
    else:
        f=open(Dataloc+fname +str(No_oth)+'3subs_'+str(trials)+'_'+str(switches)+'.txt','a')
        np.savetxt(f, y[b],  delimiter=',')
        f.close()
       

