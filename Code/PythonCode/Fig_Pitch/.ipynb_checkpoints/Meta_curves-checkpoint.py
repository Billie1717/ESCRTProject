#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 08:28:29 2021

@author: billiemeadowcroft
"""
import matplotlib.pyplot as plt
import script_det_numba
import numpy as np

#Dataloc = str('/Users/billiemeadowcroft/Documents/PHD/Code/My_NumbaCurve/Data/BindSplit/')
#fname = str('bindsplit')
#Dataloc = str('/storage/users/bmeadowcroft/Scratch/scripts/DoSplots/Data/Energy/')
Dataloc = str('/storage/users/bmeadowcroft/Scratch/scripts/HalfLife/Data/')
#fname = str('Energies12')
#fname = str('HalfLife')
fname = str('HalfLifeSameBind3')
J= 20

t_max = 3.0

Curve0 = 2.0
Curve1= Curve0#2.5
Curve2= Curve0#8.0 #Curvatures
k_0= 0.8
k_1= 0.8
k_2 = 0.8
k_mem = 2.5 # bending rigidities
pitch0 = 0.0
pitch1 = 1.0
pitch2 = 4.0 # pitch/tilt
BindMem0 = -0.8
BindMem1 = BindMem0*3.2
BindMem2 = BindMem0*4.0 #bindinf energies

for j in range(J):
    #k_mem = (j+1)*2.5*(0.2)
    print(j)
    pitch1 = (j)*0.2

    kwargs = dict(seed=np.random.randint(0,150),rand_array=None,replicas=300,lattice_length=42,t_max=t_max,
                  Curve0=Curve0, Curve1=Curve1, Curve2=Curve2,k_0=k_0,k_1=k_1,k_2=k_2,k_mem=k_mem,
                  pitch0=pitch0,pitch1=pitch1,pitch2=pitch2,BindMem0=BindMem0,BindMem1=BindMem1,BindMem2=BindMem2)
        #res_python=script_det_numba.simulate(params)
        #res_numba = script_det_numba.simulate_numba(params)
    params = script_det_numba.Params(**kwargs)
    print(params)
        #y=script_det_numba.main(params)[0] #pure python version

        #params = script_det_numba.Params(**kwargs)
    y=script_det_numba.main_numba(params)[0] #numba version
#-------------------------------#
    #for q in range(3):
    #    if q ==0:
    #        f=open(Dataloc+fname +'_'+str(j)+'.txt','w')
    #        np.savetxt(f, y[q],  delimiter=',', header=str(params))
    #        f.close()
    #    elif q ==1:
    #        f=open(Dataloc+fname  +'_'+str(j)+'.txt','a')
    #        np.savetxt(f, y[q], delimiter=',')
    #        f.close()
    #    else:
    #        f=open(Dataloc+fname  +'_'+str(j)+'.txt','a')
    #        np.savetxt(f, y[q], delimiter=',')
    #        f.write("\n")
    #        f.close()
