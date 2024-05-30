#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 7  5 16:53:12 2021

@author: billiemeadowcroft
"""

import numpy as np
import numba
from numba.extending import register_jitable
from typing import NamedTuple, Tuple


class Params(NamedTuple):
    trials: int = 10  # number of simulations to average over
    switches: int = 10  # length of lattice
    No_oth:int = 10
    Curve0: float = 8.0
    Curve1: float = 2.5
    Curve2: float = 1.0
    k_0: float = 0.8  # bending rigidities
    k_1: float = 0.8
    k_2: float = 0.8
    k_mem: float = 2.5
    pitch0: float = 1.75#ch/tilt
    pitch1: float = 1.0
    pitch2: float = 0.0
    BindMem0: float = -9.6
    BindMem1: float = -2.56
    BindMem2: float = -0.8

class Res(NamedTuple):
    out: Tuple
    
    
@register_jitable
def summarize(
    params: Params,
    res: Res,
):
    (
        ERecord
    ) = res.out
    
    N = int(42)
    switches = params.switches
    trials = params.trials
    Number = switches - int(0.2*switches)
    ERecAv = np.zeros((N,N))
    for No0s in range(N):
            for No1s in range(N): 
                for j in range(Number):
                    for s in range(trials):
                        ERecAv[No0s][No1s] += ERecord[No1s][No0s][j][s]/(Number*trials)
    zeroth = ERecAv[0][0]
    return ERecAv, zeroth

@register_jitable
def simulate(
    params: Params,
):

# =============================================================================
# #system parameters
# =============================================================================

    k_0 = params.k_0 # bending rigidities
    k_1 = params.k_1
    k_oth = params.k_2
    k_mem = params.k_mem
    Curve0 = params.Curve0 
    Curve1 = params.Curve1
    Curve_oth = params.Curve2
    pitch0 = params.pitch0  # pitch/tilt
    pitch1 = params.pitch1
    pitch_oth = params.pitch2
    
    perc = 0.0
    
    E_bind0 = (params.BindMem0)*perc
    BindMem0 = (params.BindMem0)*(1-perc)
    E_bind1 = (params.BindMem1)*perc
    BindMem1 = (params.BindMem1)*(1-perc)
    E_bind_oth = (params.BindMem2)*perc
    BindMem_oth = (params.BindMem2)*(1-perc)
    
    Curve01 = (1/(k_0+k_1))*(k_0*Curve0+k_1*Curve1)
    Curve02 = (1/(k_oth+k_0))*(k_oth*Curve_oth+k_0*Curve0)
    Curve12 = (1/(k_oth+k_1))*(k_oth*Curve_oth+k_1*Curve1)
    Curve012 = (1/(k_0+k_1+k_oth))*(k_0*Curve0+k_1*Curve1+k_oth*Curve_oth)
    
    pitch01 = (1/(k_0+k_1))*(k_0*pitch0+k_1*pitch1)
    pitch02 = (1/(k_0+k_oth))*(k_0*pitch0+k_oth*pitch_oth)
    pitch12 = (1/(k_oth+k_1))*(k_oth*pitch_oth+k_1*pitch1)
    pitch012 = (1/(k_0+k_1+k_oth))*(k_0*pitch0+k_1*pitch1+k_oth*pitch_oth)
    
    
    
    
    
    # =============================================================================
    # #simulation parameters
    # =============================================================================
    N_neigh = 4 #neighbours to average local curvature
    N = 42 #length of lattice
    
    

    No_oth = params.No_oth
    No0s_ = N-1
    No1s_ = N-1
    switches = params.switches
    trials = params.trials#no of lattice sites we're trying
    aa=0
    at=0
    Number = switches - int(0.2*switches)
    EtotAv = np.zeros((N,N, switches))
    ERecord = np.zeros((N,N, Number, trials))
    Diff =np.zeros(switches)
    def MakeLattice(No1s_,No0s_, No_oth):
        Lattice = np.ones(N)*7
        # =============================================================================
        # #initialising a lattice with a random configuration of No1s 1s
        # =============================================================================
        Count0 = 0
        Count1 = 0
        Count_oth = 0
        while Count0 < No0s_:
            for site in range(N):
                r1 = np.random.random()
                if Count0 < No0s_:
                    if r1 < 1/N:
                        if Lattice[site] == 7:
                            Lattice[site] = 0
                            Count0 +=1
                        elif Lattice[site] == 2:
                            Lattice[site] = 1
                            Count0 +=1
                        elif Lattice[site] == 3:
                            Lattice[site] = 6
                            Count0 +=1
                        elif Lattice[site] == 4:
                            Lattice[site] = 5
                            Count0 +=1
        while Count1 < No1s_:
            for site in range(N):
                r1 = np.random.random()
                if Count1 < No1s_:
                    if r1 < 1/N:
                        if Lattice[site] == 7:
                            Lattice[site] = 2
                            Count1 += 1
                        elif Lattice[site] == 0:
                            Lattice[site] = 1               
                            Count1 += 1 
                        elif Lattice[site] == 4:
                            Lattice[site] = 3               
                            Count1 += 1 
                        elif Lattice[site] == 5:
                            Lattice[site] = 6               
                            Count1 += 1 
        while Count_oth < No_oth:
           for site in range(N):
               r1 = np.random.random()
               if Count_oth < No_oth:
                   if r1 < 1/N:
                       if Lattice[site] == 7:
                           Lattice[site] = 4
                           Count_oth += 1
                       elif Lattice[site] == 0:
                           Lattice[site] = 5              
                           Count_oth += 1 
                       elif Lattice[site] == 1:
                           Lattice[site] = 6               
                           Count_oth += 1 
                       elif Lattice[site] == 2:
                           Lattice[site] = 3               
                           Count_oth += 1 
        return Lattice
    
    
    def BendingTake(LocCurve, site, c,Lattice, CurveSub, CurveNoSub):
    
        Ebend_takeaway = 0
            
        for i in range(N):        
            LocCurveAbs = LocCurve[i]
            
            if Lattice[i] == 0:
                Ebend_takeaway += (1 / 2) * k_0 * (LocCurveAbs - Curve0)**2
            elif Lattice[i] == 1:
                Ebend_takeaway += (1 / 2) * k_0 * (LocCurveAbs - Curve0)**2 + (
                    1 / 2
                ) * k_1 * (LocCurveAbs - Curve1)**2
            elif Lattice[i] == 2:
                Ebend_takeaway += (1 / 2) * k_1 * (LocCurveAbs - Curve1)**2
            elif Lattice[i] == 3:
                Ebend_takeaway += (1 / 2) * k_oth * (LocCurveAbs - Curve_oth)**2 + (
                    1 / 2
                ) * k_1 * (LocCurveAbs - Curve1)**2
            elif Lattice[i] == 4:
                Ebend_takeaway += (1 / 2) * k_oth * (LocCurveAbs - Curve_oth)**2
            elif Lattice[i] == 5:
                Ebend_takeaway += (1 / 2) * k_0 * (LocCurveAbs - Curve0)**2 + (
                    1 / 2
                ) * k_oth * (LocCurveAbs - Curve_oth)**2
            elif Lattice[i] == 6:
                Ebend_takeaway += (
                    (1 / 2) * k_0 * (LocCurveAbs - Curve0)**2
                    + (1 / 2) * k_1 * (LocCurveAbs - Curve1)**2
                    + (1 / 2) * k_oth * (LocCurveAbs - Curve_oth)**2
                )
                
        return Ebend_takeaway
    
    
    def LocCurveFunct(Lattice):
        LocCurve = np.zeros(N)
        
        for site in range(N):
            if site <= N_neigh:
                J = site
                LimS = -J
                LimB = N_neigh
            elif site >= N-N_neigh:
                J = N-site
                LimS = -N_neigh
                LimB = J
            elif site > N_neigh and site < N-N_neigh:
                LimS = -N_neigh
                LimB = N_neigh
            
            c=0
            for n in range(LimS, LimB):
                if Lattice[site+n]== 0:
                    c += 1
                    LocCurve[site] =LocCurve[site] + Curve0
                if Lattice[site+n]== 1:
                    c+= 1
                    LocCurve[site] = LocCurve[site] + Curve01
                if Lattice[site+n]== 2:
                    c+=1
                    LocCurve[site] = LocCurve[site] + Curve1
                if Lattice[site+n]== 3:  
                    c+=1
                    LocCurve[site] = LocCurve[site] + Curve12                      
                if Lattice[site+n]== 4:
                    c+=1
                    LocCurve[site] = LocCurve[site] + Curve_oth
                if Lattice[site+n]== 5:
                    c+=1
                    LocCurve[site] = LocCurve[site] + Curve02
                if Lattice[site+n]== 6:
                    c+=1
                    LocCurve[site] = LocCurve[site] + Curve012
            if c!= 0:
                LocCurve[site] = LocCurve[site]/c 
        return LocCurve
    
    def ComputeEnergies(Lattice):
            
        GlobCurve  = 0
        GlobPitch = 0
        g = 0
        Ebind = 0
        for l in range(N):
            if Lattice[l]==0:
                GlobCurve += Curve0
                GlobPitch += pitch0  
                Ebind += BindMem0                      
                g += 1
            elif Lattice[l]==1:
                GlobCurve += Curve01
                GlobPitch += pitch01  
                Ebind += BindMem0  + BindMem1 + E_bind0 + E_bind1             
                g+=1
            elif Lattice[l]==2:
                GlobCurve += Curve1
                GlobPitch += pitch1  
                Ebind += BindMem1                     
                g+=1  
            elif Lattice[l]==3:
                GlobCurve += Curve12
                GlobPitch += pitch12  
                Ebind += BindMem_oth  + BindMem1 + E_bind_oth + E_bind1             
                g+=1   
            elif Lattice[l]==4:
                GlobCurve += Curve_oth
                GlobPitch += pitch_oth
                Ebind += BindMem_oth                    
                g+=1
            elif Lattice[l]==5:
                GlobCurve += Curve02
                GlobPitch += pitch02  
                Ebind += BindMem_oth  + BindMem0 + E_bind_oth + E_bind0          
                g+=1
            elif Lattice[l]==6:
                GlobCurve += Curve012
                GlobPitch += pitch012  
                Ebind += BindMem_oth  + BindMem0 + E_bind_oth + E_bind0 + E_bind1 + BindMem1      
                g+=1      
                 
        GlobPitch = GlobPitch/N
        if g != 0:
            GlobCurve = GlobCurve/g
            
        if GlobCurve == 0:
            Emem = 0
        else:
            Emem = 8*np.pi*k_mem*(GlobPitch**2/(GlobCurve**-2+GlobPitch**2))
            if GlobPitch > (GlobCurve)**-1:
                Emem = np.pi*k_mem*(GlobCurve*GlobPitch-1+4)
        
        
        Etot = BendingTake(LocCurveFunct(Lattice), 1, 1,Lattice, 1, 1)  + Ebind + Emem
        
        return Etot
    
    # =============================================================================
    # #Starting simultation
    # 
    # =============================================================================
   
    for No1s in range(No1s_):
        for No0s in range(No0s_):
            for S in range(trials):
                Lattice = MakeLattice(No1s, No0s, No_oth)
                       
                
                EtotAv[No1s][No0s][0] = ComputeEnergies(Lattice)
                for m in range(1,switches):
                    site0 = np.zeros(No0s)
                    site1 = np.zeros(No1s)
                    siteOth = np.zeros(No_oth)
                    siteEmp0 = np.zeros(N-No0s+1) #don't know why this has to be 1 bigger than I thought
                    siteEmp1 = np.zeros(N-No1s+1)
                    siteEmpOth = np.zeros(N-No_oth+1)
                    i = 0
                    j = 0
                    k = 0
                    l = 0
                    io = 0
                    ko = 0
                    for site in range(N):
                        
                        if Lattice[site] == 0 or Lattice[site] == 1 or Lattice[site] == 5 or Lattice[site] == 6:
                            site0[i] = site
                            i +=1
                        if Lattice[site] == 1 or Lattice[site] == 2 or Lattice[site] == 3 or Lattice[site] == 6:
                            site1[j] = site
                            j += 1                       
                        if Lattice[site] == 3 or Lattice[site] == 4 or Lattice[site] == 5 or Lattice[site] == 6:
                            siteOth[io] = site
                            io += 1
                        if Lattice[site] == 7 or Lattice[site] == 2 or Lattice[site] == 3 or Lattice[site] == 4:
                            siteEmp0[k] = site
                            k += 1
                        if Lattice[site] == 7 or Lattice[site] == 0 or Lattice[site] == 4 or Lattice[site] == 5:
                            siteEmp1[l] = site
                            l += 1
                        if Lattice[site] == 7 or Lattice[site] == 0 or Lattice[site] == 1 or Lattice[site] == 2:
                            siteEmpOth[ko] = site
                            ko += 1
                            
                    r1 = np.random.random()
                    r2 = np.random.random()
                    r4 = np.random.random()
                    r5 = np.random.random()
                    r6 = np.random.random()
                    r7 = np.random.random()
                    switch0 = int(np.floor(r1*No0s))
                    switchEmp0 = int(np.floor(r2*(N-No0s)))
                    switch1 = int(np.floor(r4*No1s))
                    switchEmp1 = int(np.floor(r5*(N-No1s)))
                    switchOth = int(np.floor(r6*No_oth))
                    switchEmpOth = int(np.floor(r7*(N-No_oth)))
                    LatticeS = np.copy(Lattice)
                    if No0s!=0:
                        for site in range(N):
                            if site == siteEmp0[switchEmp0]:
                                if LatticeS[site] == 7:
                                    LatticeS[site] = 0
                                elif LatticeS[site] == 2:
                                    LatticeS[site] = 1
                                elif LatticeS[site] == 3:
                                    LatticeS[site] = 6
                                elif LatticeS[site] == 4:
                                    LatticeS[site] = 5
                                else:
                                    print("something's gone wrong 1")
                            if site == site0[switch0]:
                                if LatticeS[site] == 0:
                                    LatticeS[site] = 7
                                elif LatticeS[site] == 1:
                                    LatticeS[site] = 2
                                elif LatticeS[site] == 6:
                                    LatticeS[site] = 3
                                elif LatticeS[site] == 5:
                                    LatticeS[site] = 4
                                else:
                                    print("something's gone wrong 2")
                    if No1s!=0:
                        for site in range(N):
                            if site == site1[switch1]:
                                if LatticeS[site] == 2:
                                    LatticeS[site] = 7
                                elif LatticeS[site] == 1:
                                    LatticeS[site] = 0             
                                elif LatticeS[site] == 3:
                                    LatticeS[site] = 4             
                                elif LatticeS[site] == 6:
                                    LatticeS[site] = 5             
                                else:
                                    print("something's gone wrong 3")
                            if site == siteEmp1[switchEmp1]:
                                if LatticeS[site] == 7:
                                    LatticeS[site] = 2
                                elif LatticeS[site] == 0:
                                    LatticeS[site] = 1               
                                elif LatticeS[site] == 4:
                                    LatticeS[site] = 3               
                                elif LatticeS[site] == 5:
                                    LatticeS[site] = 6     
                                else:
                                    print("something's gone wrong 4")
                    if No_oth!=0:
                        for site in range(N):
                            if site == siteOth[switchOth]:
                                if LatticeS[site] == 4:
                                   LatticeS[site] = 7
                                elif LatticeS[site] == 5:
                                    LatticeS[site] = 0            
                                elif LatticeS[site] == 6:
                                    LatticeS[site] = 1             
                                elif LatticeS[site] == 3:
                                    LatticeS[site] = 2             
                                else:
                                    print("something's gone wrong 5")
                            if site == siteEmpOth[switchEmpOth]:
                                if LatticeS[site] == 7:
                                   LatticeS[site] = 4
                                elif LatticeS[site] == 0:
                                    LatticeS[site] = 5              
                                elif LatticeS[site] == 1:
                                    LatticeS[site] = 6               
                                elif LatticeS[site] == 2:
                                    LatticeS[site] = 3     
                                else:
                                    print("something's gone wrong 6")
                                
                    Diff[m] = ComputeEnergies(LatticeS) - ComputeEnergies(Lattice)#EtotAv[No1s][No0s][m-1]
                    r3 = np.random.random()
                    EtotAv[No1s][No0s][m] = ComputeEnergies(LatticeS) #record trial whether its been accepted or not
                    at+=1
                    if r3 < (np.exp(-Diff[m])):
                        aa+=1
                        Lattice = np.copy(LatticeS) #accepting new config
                    
                    if m > (int(0.2*switches)):
                        n = m -int(0.2*switches)
                        ERecord[No1s][No0s][n][S] = EtotAv[No1s][No0s][m]
    acc = aa/at
                
    return Res(
    out=(
        ERecord 
    ),
    )

simulate_numba = numba.njit(simulate)


def main(params: Params):

    res = simulate(params)

    summary = summarize(params, res)

    return summary


main_numba = numba.njit(main, cache=True)

        
            


 
    