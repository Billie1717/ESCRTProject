#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 08:28:29 2021

@author: billiemeadowcroft
"""
import matplotlib.pyplot as plt
import script_det_numbaVPS4
import numpy as np

Dataloc =str('/Users/billie/Documents/PHD/Code/My_NumbaCurve/Data/VPS4/ys/')
#Dataloc = str('/Users/billiemeadowcroft/Documents/PHD/Code/My_NumbaCurve/Data/VPS4/')
fname = str('E_0v')

I= 30
J= 1

t_max = 6.0

Curve0 = 1.0
Curve1= 2.5
Curve2=8.0 #Curvatures
k_0= 0.8
k_1= 0.8
k_2 = 0.8
k_mem = 2.5 # bending rigidities
pitch0 = 0.0
pitch1 = 1.0
pitch2 = 2.0 # pitch/tilt
BindMem0 = -0.8
BindMem1 = BindMem0*3.2
BindMem2 = BindMem0*11.5 #bindinf energies
E_v = 0#7
Th = 0.45

N=42
kwargs = dict(seed=np.random.randint(0,100),rand_array=None,replicas=150,lattice_length=42,t_max=t_max,
                      Curve0=Curve0, Curve1=Curve1, Curve2=Curve2,k_0=k_0,k_1=k_1,k_2=k_2,k_mem=k_mem,
                       pitch0=pitch0,pitch1=pitch1,pitch2=pitch2,BindMem0=BindMem0,BindMem1=BindMem1,BindMem2=BindMem2, E_v = E_v, Th = Th)
params = script_det_numbaVPS4.Params(**kwargs)
print(params)
y=script_det_numbaVPS4.main_numba(params)[0]

fig,ax = plt.subplots()
ax.plot(y[0]*100/N)
ax.plot(y[1]*100/N)
ax.plot(y[2]*100/N)

for i in range(I):
    for kk in range(J):
        k = kk+3 
        k_0 = (i+1)*0.05
        k_1 = (i+1)*0.05
        k_2 = (i+1)*0.05
        E_v = 0#(k+1)*1.625


        kwargs = dict(seed=np.random.randint(0,100),rand_array=None,replicas=150,lattice_length=42,t_max=t_max,
                      Curve0=Curve0, Curve1=Curve1, Curve2=Curve2,k_0=k_0,k_1=k_1,k_2=k_2,k_mem=k_mem,
                      pitch0=pitch0,pitch1=pitch1,pitch2=pitch2,BindMem0=BindMem0,BindMem1=BindMem1,BindMem2=BindMem2, E_v = E_v, Th = Th)
        #res_python=script_det_numba.simulate(params)
        #res_numba = script_det_numba.simulate_numba(params)
        params = script_det_numbaVPS4.Params(**kwargs)
        print(params)
        #y=script_det_numba.main(params)[0] #pure python version

        #params = script_det_numba.Params(**kwargs)
        y=script_det_numbaVPS4.main_numba(params)[0] #numba version

        for q in range(3):
            if q ==0:
                f=open(Dataloc+fname+ str(k)+'_'+ str(i) +'.txt','w')
                np.savetxt(f, y[q],  delimiter=',', header=str(params))
                f.close()
            elif q ==1:
                f=open(Dataloc+fname+ str(k)+'_' + str(i) +'.txt','a')
                np.savetxt(f, y[q], delimiter=',')
                f.close()
            else:
                f=open(Dataloc+fname+ str(k)+'_' + str(i) +'.txt','a')
                np.savetxt(f, y[q], delimiter=',')
                f.write("\n")
                f.close()
