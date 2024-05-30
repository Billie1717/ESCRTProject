#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 16:55:23 2020

@author: billiemeadowcroft
"""

import numpy as np
import random
import pandas
from typing import NamedTuple, Optional, Tuple

import numba
from numba.extending import register_jitable
class Params(NamedTuple):
    y: np.ndarray = None

@register_jitable
#@numba.jit(nopython=True)
def DataProcess(
    params: Params,
):

    y = params.y

    
    L = len(y[0])
    
    # =============================================================================
    # #Finding t_max
    # =============================================================================
    
    maxProb0 = max(y[0])
    maxProb1 = max(y[1])
    maxProb2 = max(y[2])
    halfmax0 = 0
    halfmax1 = 0
    halfmax2 = 0
    for i in range(L):
        if y[0][i] == max(y[0]):
            t_0 = i 
        if y[1][i] == max(y[1]):
            t_1 = i 
        if y[2][i] == max(y[2]):
            t_2 = i
        if abs(y[0][i]-(max(y[0]))/2) < abs(halfmax0-max(y[0])/2):
            halfmax0 = y[0][i]
            thalf0 = i
        if abs(y[1][i]-(max(y[1]))/2) < abs(halfmax1-max(y[1])/2):
            halfmax1 = y[1][i]
            thalf1 = i
        if abs(y[2][i]-(max(y[2]))/2) < abs(halfmax2-max(y[2])/2):
            halfmax2 = y[2][i]
            thalf2 = i
    
    
    
    LastProb0 = y[0][L-1]
    LastProb1 = y[1][L-1]
    LastProb2 = y[2][L-1]
    P1t0 = y[1][t_0]
    P2t0 = y[2][t_0]
    P0t1 = y[0][t_1]
    P2t1 = y[2][t_1]
    
    # =============================================================================
    # #Finding FWTQM
    # =============================================================================
    
    # =============================================================================
    # #Finding FWTQM
    # =============================================================================
    return maxProb0, maxProb1, maxProb2, LastProb0, LastProb1, LastProb2, P1t0, P2t0, P0t1, P2t1
               

DataProcess_numba = numba.njit(DataProcess)