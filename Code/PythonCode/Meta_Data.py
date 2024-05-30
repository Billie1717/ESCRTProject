#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 08:28:29 2021

@author: billiemeadowcroft
"""
import matplotlib.pyplot as plt
import Data_Proc
import numpy as np
import pandas


#Dataloc = str('/Users/billiemeadowcroft/Documents/PHD/Code/My_NumbaCurve/Data/BindSplit/')
#fname = str('bindsplit')
#datadir = str('/Users/billiemeadowcroft/Documents/PHD/Code/My_NumbaCurve/Data/Pitch12/')
#filename = str('pitch12')
L = 1
J= 30
#datadir = str('/Users/billiemeadowcroft/Documents/PHD/Code/My_NumbaCurve/Data/MemRig/')
#filename = str('MemRig')
#Dataloc = str('/storage/users/bmeadowcroft/Numba_Torsion/Data/TorsCur/')
Dataloc = str('/Users/billie/Documents/PHD/ESCRT_Proj1/Curvy_model/GitPlots/MechanoChemDataAndPlotting/SIplots/FigsLattice_ronroff/Data/')
datadir = str('/Users/billie/Documents/PHD/ESCRT_Proj1/Curvy_model/GitPlots/MechanoChemDataAndPlotting/SIplots/FigsLattice_ronroff/Data/')

#datadir =str('/Users/billie/Documents/PHD/Code/My_NumbaCurve/Data/VPS4/')
#filename = str('Lattice_Evar')
#fname = str('Lattice_Evar')
#filename = str('Lattice_neighvar')
#fname = str('Lattice_neighvar')
#filename = str('Lattice')
#fname = str('Lattice')
fname = str('Ron')
filename = str('Ron')
Ns = [42,84,126,168,210] #,168,210]
for l in range(L):
    #for j in range(J):
    N = Ns[l]
    q = 3200 #8000 #24000
    #yList = pandas.read_csv(datadir +'ys/'+ filename + str(l)+ '_'+str(j) +".txt",header = 'infer', sep='\n')
    P0 = pandas.read_csv(datadir +fname+ str(N)+'Prob0.txt',header = 'infer', sep='\n')
    P1 = pandas.read_csv(datadir +fname+ str(N)+'Prob1.txt',header = 'infer', sep='\n')
    P2 = pandas.read_csv(datadir +fname+ str(N)+'Prob2.txt',header = 'infer', sep='\n')
    y1=np.array([(42/N)*P0,(42/N)*P1,(42/N)*P2]) 
    y = y1.reshape(3,q-1)
    
    #print('shape y',np.shape(y))#yList.values.reshape(3,q)
    kwargs = dict(y = y)
    params = Data_Proc.Params(**kwargs)
    Chars = Data_Proc.DataProcess_numba(params)
    f = open(datadir+filename+"CharCrv.txt", "a")
    f.write('{}'.format(Chars)+' \n')
    f.close()
    fig,ax = plt.subplots()
    ax.plot(y[0], label = 'mon 0', color = 'lightblue')
    ax.plot(y[1], label = 'mon 1', color = 'mediumpurple')
    ax.plot(y[2], label = 'mon 2', color = 'pink')
    print('max0', max(y[0]))
    print('max0', max(P0))
