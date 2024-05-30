#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 08:28:29 2021

@author: billiemeadowcroft
"""
import matplotlib.pyplot as plt
import script_tors2
import numpy as np


Dataloc = str('/Users/billiemeadowcroft/Documents/PHD/Code/My_NumbaCurve/Data/RouxFitting/')
Dataloc = str('/Users/billiemeadowcroft/Documents/PHD/Code/My_NumbaCurve/Data/MemRig_vs_binding/')
fname= str('Torsv2')

t_max = 1.0

Curve0 = 1.0 #1.0
Curve1 = 2.5#2.5
Curve2 = 8.0#8.0
k_0 = 0.8 
k_1 = k_0
k_2 = k_0
k_mem = 2.5 #3.0 # bending rigidities
P0 = 0.0
P1 = 0.1 #gives a total length helix of 0.5 with a length of 1000nm with 0.785 0.09
P2 = 0.095 # pitch/tilt 0.095
BindMem0 = -0.8
BindMem1 = -2.5
BindMem2 = -9.5 #binding energies
ktFactor = 0.11
R0 = 1/Curve0
R1 = 1/Curve1
R2 = 1/Curve2
N=42
length = 10 #20

# =============================================================================
# t_max = 1.0
# 
# Curve1 = 1.0#2.0
# Curve0 = 1.0 #0.8
# Curve2 = 1.0#9.1
# k_0 = 16 #1.6
# k_1 = k_0/10
# k_2 = k_0/10
# k_mem = 3.5 # bending rigidities 7.5
# P0 = 0.0
# P1 = 0.09 #gives a total length helix of 0.5 with a length of 1000nm with 0.785 0.09
# P2 = 0.3# pitch/tilt 0.095
# BindMem0 = -0.1 #0.08
# BindMem1 = -2.3#2.3
# BindMem2 = -9.5 #binding energies
# ktFactor = 1.0 #0.1 #ratio between torsion and bending rigidity
# R0 = 1/Curve0
# R1 = 1/Curve1
# R2 = 1/Curve2
# N=42
# length = 30 #20
# =============================================================================
turns1 = length/(np.sqrt((2*np.pi*R1)**2+(P1)**2))
Pitch1 = P1*turns1
turns2 = length/(np.sqrt((2*np.pi*R2)**2+(P2)**2))
Pitch2 = P2*turns2


print("Pitch1:", Pitch1, "Pitch2:", Pitch2)
# =============================================================================
# for i in range(I):
#     ktFact = i*0.1
#     for j in range(J):
# 
#         Curve1 = Curve0 + 0.5 - j*0.02
#         Curve2 = Curve0 + 5 - j*0.2
# =============================================================================


kwargs = dict(seed=np.random.randint(0,100),rand_array=None,replicas=30,lattice_length=42,t_max=t_max,
                                  R0=R0, R1=R1, R2=R2,k_0=k_0,k_1=k_1,k_2=k_2,k_mem=k_mem,
                                  P0=P0,P1=P1,P2=P2,BindMem0=BindMem0,BindMem1=BindMem1,BindMem2=BindMem2, ktFactor = ktFactor, length=length)
                    #res_python=script_det_numba.simulate(params)
                    #res_numba = script_det_numba.simulate_numba(params)
params = script_tors2.Params(**kwargs)
print(params)
                    #y=script_det_numba.main(params)[0] #pure python version
            
                    #params = script_det_numba.Params(**kwargs)
y=script_tors2.main_numba(params)[0] #numba version
    
# =============================================================================
# for q in range(3):
#     if q ==0:
#         f=open(Dataloc+fname+"MemRig"+str(k_mem)+'Prob0.txt','w')
#         np.savetxt(f, y[q],  delimiter=',', header=str(params))
#         f.close()
#     elif q ==1:
#         f=open(Dataloc+fname +"MemRig"+str(k_mem)+'Prob1.txt','w')
#         np.savetxt(f, y[q], delimiter=',')
#         f.close()
#     else:
#         f=open(Dataloc+fname +"MemRig"+str(k_mem)+'Prob2.txt','w')
#         np.savetxt(f, y[q], delimiter=',')
#         f.write("\n")
#         f.close()
# =============================================================================
fig,ax = plt.subplots()
ax.plot(y[0]*100/N, label = "0")
ax.plot(y[1]*100/N, label = "1")
ax.plot(y[2]*100/N, label = "2")
plt.text(30,35,'C0,C1,C2:'+str(Curve0)+','+str(Curve1)+','+str(Curve2)+' '+'KFact:' + str(ktFactor))
ax.legend()
# =============================================================================
# fig2,ax2 = plt.subplots()
# ax2.plot(loccurveav, label = "LocCurveAv")
# ax2.legend()
# =============================================================================
