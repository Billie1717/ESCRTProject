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
L = 1#20
J= 20#20
#datadir = str('/Users/billiemeadowcroft/Documents/PHD/Code/My_NumbaCurve/Data/MemRig/')
#filename = str('MemRig')
#datadir = str('/storage/users/bmeadowcroft/Scratch/scripts/DoSplots/Data/Energy/')
#filename = str('Energies12')
#datadir = str('/storage/users/bmeadowcroft/Scratch/scripts/DoSplots/Data/Curvature/')
#filename = str('Curves12_')
datadir = str('/storage/users/bmeadowcroft/Scratch/scripts/BindSplit/Data/')
filename = str('BindSplit2_')
for l in range(L):
    for j in range(J):
        q = 24000
        yList = pandas.read_csv(datadir + filename +  '_'+str(j) +".txt",header = 'infer', sep='\n')
        y=yList.values.reshape(3,q)
        kwargs = dict(y = y)
        params = Data_Proc.Params(**kwargs)
        Chars = Data_Proc.DataProcess_numba(params)
        f = open(datadir+filename+"CharCrv.txt", "a")
        f.write('{}'.format(Chars)+' \n')

