#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 08:28:29 2021

@author: billiemeadowcroft
"""
import matplotlib.pyplot as plt
import script_det_numba
#import script_det_numbaVPS4
import numpy as np

#Dataloc = str('/Users/billiemeadowcroft/Documents/PHD/Code/My_NumbaCurve/Data/MemRig_vs_binding//')
#Dataloc = str('/Users/billiemeadowcroft/Documents/PHD/Curvy_model/GitPlots/MechanoChemDataAndPlotting/Fig_Arrows/CorrectedData/')
#Dataloc = str('/Users/billiemeadowcroft/Documents/PHD/Curvy_model/GitPlots/MechanoChemDataAndPlotting/Fig_MemDef/Data/')
#Dataloc = str('/Users/billiemeadowcroft/Documents/PHD/Curvy_model/GitPlots/MechanoChemDataAndPlotting/Fig_EnergyBarrier/Data/')
#Dataloc = str('/Users/billie/Documents/PHD/Curvy_model/GitPlots/MechanoChemDataAndPlotting/SIplots/BindSplitting/')
Dataloc = str('//Users/billiemeadowcroft/Dropbox/Code/ESCRTfresh/My_NumbaCurve/TestingCode/Data/')
#Dataloc = str('/Users/billie/Documents/PHD/ESCRT_Proj1/Curvy_model/Curve_plotsReDone/FigLattice/Data/')
#Dataloc = str('/Users/billiemeadowcroft/Documents/PHD/Curvy_model/GitPlots/MechanoChemDataAndPlotting/Fig_DoS/')
#Dataloc =str('/Users/billie/Documents/PHD/Code/My_NumbaCurve/Data/VPS4/ys/')
#Dataloc = str('/Users/billiemeadowcroft/Documents/PHD/Code/My_NumbaCurve/Data/VPS4/')
fname = str('bigspaceTest1_')
fname = str('BindSplitStar')
fname = str('BindSplitSquare')
fname = str('Lattice')

I=30
t_max = 1.0

Curve0 = 1.0 #1.0
Curve1 = Curve0*2.5    #2.5
Curve2 = Curve0*8.0    #Curvatures #8.0
k_0 = 0.8
k_1 = k_0
k_2 = k_0
k_mem = 2.5 # bending rigidities #2.5
pitch0 = 0.0 #0.1
pitch1 = 1.0 #1.75 #1.0
pitch2 = 1.5 #2.0 # pitch/tilt #1.5
BindMem0 = -0.8 #-0.08
BindMem1 = BindMem0*3   #-2.3
BindMem2 = BindMem0*12 #11.5     #12 #-9.5
FactorKa = np.exp(1)
FactorKoff = np.exp(1)
N=42 #126
N_neigh = 4
#fname = 'Pathway1_'
#fname = 'smolspaceTest1_'
#fname = 'Arrows_c'
#fname = 'Seq'
#fname = 'CsEq_'
#fname = 'PsEq_'
#fname = '2subs_'

kwargs = dict(seed=np.random.randint(0,100),rand_array=None,replicas=100,lattice_length=N,t_max=t_max,
                      Curve0=Curve0, Curve1=Curve1, Curve2=Curve2,k_0=k_0,k_1=k_1,k_2=k_2,k_mem=k_mem,
                      pitch0=pitch0,pitch1=pitch1,pitch2=pitch2,BindMem0=BindMem0,BindMem1=BindMem1,BindMem2=BindMem2, FactorKa=FactorKa, FactorKoff=FactorKoff, N_neigh=N_neigh)
params = script_det_numba.Params(**kwargs)
print(params)
y=script_det_numba.main_numba(params)[0] #numba version
GlobPitch = script_det_numba.main_numba(params)[3]
Emem = script_det_numba.main_numba(params)[5]
Ebind = script_det_numba.main_numba(params)[6]
Ebend = script_det_numba.main_numba(params)[7]

for q in range(3): ##Decide how much you want to save!
    if q ==0:
        f=open(Dataloc+fname+ str(N)+'Prob0.txt','w')
        np.savetxt(f, y[q],  delimiter=',') #, header=str(params))
        f.close()
    elif q ==1:
        f=open(Dataloc+fname+ str(N) +'Prob1.txt','w')
        np.savetxt(f, y[q], delimiter=',')
        f.close()
    elif q==2:
        f=open(Dataloc+fname+ str(N) +'Prob2.txt','w')
        np.savetxt(f, y[q], delimiter=',')
        f.write("\n")
        f.close()
    elif q==3:
        f=open(Dataloc+fname +'Emem.txt','w')
        np.savetxt(f, Emem, delimiter=',')
        f.write("\n")
        f.close()
    elif q==4:
        f=open(Dataloc+fname +'Ebind.txt','w')
        np.savetxt(f, Ebind, delimiter=',')
        f.write("\n")
        f.close()
    else:
        f=open(Dataloc+fname +'Ebend.txt','w')
        np.savetxt(f, Ebend, delimiter=',')
        f.write("\n")
        f.close()



#fig,ax = plt.subplots()
#ax.plot(y[0]*100/N, label = 'mon 0', color = 'lightblue')
#ax.plot(y[1]*100/N, label = 'mon 1', color = 'mediumpurple')
#ax.plot(y[2]*100/N, label = 'mon 2', color = 'pink')
#fig.savefig(Dataloc+'TestPlot.png')





# =============================================================================
# for i in range(I):
#     k=3
#     k_0 = (i+1)*0.05
#     k_1 = (i+1)*0.05
#     k_2 = (i+1)*0.05
#     E_v = 0#(k+1)*1.625
# 
# 
#     kwargs = dict(seed=np.random.randint(0,100),rand_array=None,replicas=100,lattice_length=42,t_max=t_max,
#                            Curve0=Curve0, Curve1=Curve1, Curve2=Curve2,k_0=k_0,k_1=k_1,k_2=k_2,k_mem=k_mem,
#                            pitch0=pitch0,pitch1=pitch1,pitch2=pitch2,BindMem0=BindMem0,BindMem1=BindMem1,BindMem2=BindMem2, FactorKa=FactorKa, FactorKoff=FactorKoff)
# 
#     params = script_det_numba.Params(**kwargs)
#     print(params)
# 
#     y=script_det_numba.main_numba(params)[0] #numba version
# 
#     for q in range(3):
#         if q ==0:
#             f=open(Dataloc+fname+ str(k)+'_'+ str(i) +'.txt','w')
#             np.savetxt(f, y[q],  delimiter=',', header=str(params))
#             f.close()
#         elif q ==1:
#             f=open(Dataloc+fname+ str(k)+'_' + str(i) +'.txt','a')
#             np.savetxt(f, y[q], delimiter=',')
#             f.close()
#         else:
#             f=open(Dataloc+fname+ str(k)+'_' + str(i) +'.txt','a')
#             np.savetxt(f, y[q], delimiter=',')
#             f.write("\n")
#             f.close()
# =============================================================================

# =============================================================================
# for q in range(1):
#     if q ==0:
#         f=open(Dataloc+fname+ 'GlobPitch.txt','w')
#         np.savetxt(f, GlobPitch,  delimiter=',')#, header=str(params))
#         f.close()
# =============================================================================
# =============================================================================
# for q in range(3):
#     if q ==0:
#         f=open(Dataloc+fname+ str(N)+'Prob0.txt','w')
#         np.savetxt(f, y[q],  delimiter=',')#, header=str(params))
#         f.close()
#     elif q ==1:
#         f=open(Dataloc+fname+ str(N) +'Prob1.txt','w')
#         np.savetxt(f, y[q], delimiter=',')
#         f.close()
#     elif q==2:
#         f=open(Dataloc+fname+ str(N) +'Prob2.txt','w')
#         np.savetxt(f, y[q], delimiter=',')
#         f.write("\n")
#         f.close()
#     elif q==3:
#         f=open(Dataloc+fname +'Emem.txt','w')
#         np.savetxt(f, Emem, delimiter=',')
#         f.write("\n")
#         f.close()
#     elif q==4:
#         f=open(Dataloc+fname +'Ebind.txt','w')
#         np.savetxt(f, Ebind, delimiter=',')
#         f.write("\n")
#         f.close()
#     else:
#         f=open(Dataloc+fname +'Ebend.txt','w')
#         np.savetxt(f, Ebend, delimiter=',')
#         f.write("\n")
#         f.close()
# =============================================================================

