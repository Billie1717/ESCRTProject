#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar 17  5 16:53:12 2021

@author: billiemeadowcroft
"""

from typing import NamedTuple, Optional, Tuple
import numpy as np

# import matplotlib.pyplot as plt
# import statistics as stat
# import seaborn as sns

import numba
from numba.extending import register_jitable, overload


class Params(NamedTuple):
    seed: int = 0
    rand_array: Optional[np.ndarray] = None
    replicas: int = 10  # number of simulations to average over
    lattice_length: int = 42  # length of lattice
    t_max: float = 2
    Curve0: float = 1.0
    Curve1: float = 2.4
    Curve2: float = 8.0
    k_0: float = 0.8  # bending rigidities
    k_1: float = 0.8
    k_2: float = 0.8
    k_mem: float = 2.5
    pitch0: float = 0.0  # pitch/tilt
    pitch1: float = 1.0
    pitch2: float = 1.5
    BindMem0: float = -0.08
    BindMem1: float = -2.3
    BindMem2: float = -9.5
    E_v: float = 5
    Th: float = 0.4


class Res(NamedTuple):
    out: Tuple
    steps: int
    random_c: int


@register_jitable
def summarize(
    params: Params,
    res: Res,
):
    (
        Latticeline,
        LocCurveT,
        GlobPitchtot,
        GlobCurvetot,
        BendEnergyDiff,
        EmemTot,
        EbindTot,
        EbendTot,
    ) = res.out

    lattice_length = params.lattice_length
    replicas = params.replicas

    def round_down(num, divisor):
        return num - (num % divisor)


    w = round_down(
        res.steps, 10
    )  # depending on when the simulation finishes sometimes q = t_max/dt +1, just getting rid of the 1

    y = np.zeros((3, w))
    ste = np.zeros((8, w))

    x = []

    LocCurveAv = np.zeros((lattice_length, w))
    GlobPitch_time = np.zeros(w)
    GlobCurve_time = np.zeros(w)
    Emem_time = np.zeros(w)
    Ebind_time = np.zeros(w)
    Ebend_time = np.zeros(w)
    EBDiff_time = np.zeros(w)

    for j in range(w):  # lets scan over all time
        f0 = 0
        f1 = 0
        f2 = 0
        s0 = 0
        s1 = 0
        s2 = 0
        s3 = 0
        s4 = 0
        s5 = 0
        s6 = 0
        s7 = 0
        x.append(j)
        for k in range(replicas):
            GlobPitch_time[j] += GlobPitchtot[j][k] / replicas
            GlobCurve_time[j] += GlobCurvetot[j][k] / replicas
            Emem_time[j] += EmemTot[j][k] / replicas
            Ebind_time[j] += EbindTot[j][k] / replicas
            Ebend_time[j] += EbendTot[j][k] / replicas
            EBDiff_time[j] += BendEnergyDiff[j][k] / replicas

            for i in range(lattice_length):  # lets go through the lattice at each time
                if Latticeline[k][j][i] == 0:
                    f0 = f0 + 1  # frequency of sites in state 0 goes up by 1
                    s0 = s0 + 1
                elif Latticeline[k][j][i] == 1:
                    f0 = f0 + 1
                    f1 = f1 + 1
                    s1 = s1 + 1
                elif Latticeline[k][j][i] == 2:
                    f1 = f1 + 1
                    s2 = s2 + 1
                elif Latticeline[k][j][i] == 3:
                    f2 = f2 + 1
                    f1 = f1 + 1
                    s3 = s3 + 1
                elif Latticeline[k][j][i] == 4:
                    f2 = f2 + 1
                    s4 = s4 + 1
                elif Latticeline[k][j][i] == 5:
                    f0 = f0 + 1
                    f2 = f2 + 1
                    s5 = s5 + 1
                elif Latticeline[k][j][i] == 6:
                    f0 = f0 + 1
                    f1 = f1 + 1
                    f2 = f2 + 1
                    s6 = s6 + 1
                else:
                    s7 = s7 + 1
                # f0= 0,1,5,6
                # f1= 1,2,3,6
                # f2= 3,4,5,6

                LocCurveAv[i][j] += LocCurveT[k][j][i] / replicas

            y[0][j] += f0 / (0.5 * replicas * (replicas + 1))
            y[1][j] += f1 / (0.5 * replicas * (replicas + 1))
            y[2][j] += f2 / (0.5 * replicas * (replicas + 1))
            ste[0][j] += s0 / (0.5 * replicas * (replicas + 1))
            ste[1][j] += s1 / (0.5 * replicas * (replicas + 1))
            ste[2][j] += s2 / (0.5 * replicas * (replicas + 1))
            ste[3][j] += s3 / (0.5 * replicas * (replicas + 1))
            ste[4][j] += s4 / (0.5 * replicas * (replicas + 1))
            ste[5][j] += s5 / (0.5 * replicas * (replicas + 1))
            ste[6][j] += s6 / (0.5 * replicas * (replicas + 1))
            ste[7][j] += s7 / (0.5 * replicas * (replicas + 1))

    return (
        y,
        ste,
        LocCurveAv,
        GlobPitch_time,
        GlobCurve_time,
        Emem_time,
        Ebind_time,
        Ebend_time,
        EBDiff_time,
    )


@register_jitable
def summarize2endstate(
    y,
    ste,
    LocCurveAv,
    GlobPitch_time,
    GlobCurve_time,
    Emem_time,
    Ebind_time,
    Ebend_time,
    EBDiff_time,
):
    return y[0][-1], y[1][-1], y[2][-1]


@register_jitable

def simulate(
    params: Params,
):

    # random = np.random.default_rng(seed=seed) # this would work for python, but not for numba. we use the global generator instead

    np.random.seed(params.seed)
    random_c = 0

    # little dance to deal with Optional[np.ndarray] not working exactly right with numba
    # and to satisfy the static analyzer
    maybe_rand_array = params.rand_array
    has_rand_array = maybe_rand_array is not None
    _rand_array = maybe_rand_array if maybe_rand_array is not None else np.empty((0,))

    def random():
        nonlocal random_c
        if has_rand_array:
            assert random_c < len(_rand_array)
            r = _rand_array[random_c]
        else:
            r = np.random.random()
        random_c += 1
        return r

    # =============================================================================
    # #system parameters
    # =============================================================================

    conc0 = 2.0  # concentration
    conc1 = 1.0
    conc2 = 1.0
    ka0 = 0.8  # association rate
    ka1 = 0.8
    ka2 = 0.8
    Kon0 = ka0 * conc0  # Kon (constant through time)
    Kon1 = ka1 * conc1
    Kon2 = ka2 * conc2
    k_0 = params.k_0 # bending rigidities
    k_1 = params.k_1
    k_2 = params.k_2
    k_mem = params.k_mem
    Curve0 = params.Curve0 
    Curve1 = params.Curve1
    Curve2 = params.Curve2
    pitch0 = params.pitch0  # pitch/tilt
    pitch1 = params.pitch1
    pitch2 = params.pitch2
    BindMem0 = params.BindMem0
    BindMem1 = params.BindMem1
    BindMem2 = params.BindMem2
    BindFil0 = 0.0#- 0.08 - BindMem0 #0.0  # -0.015
    BindFil1 = 0.0##-2.3 - BindMem1  # -0.345
    BindFil2 = 0.0#-9.5- BindMem2  # -1.425
    E_v = params.E_v
    Th = params.Th
    
    # =============================================================================
    # #simulation parameters
    # =============================================================================
    N_neigh = 4  # neighbours to average local curvature
    # N = 42
    lattice_length = params.lattice_length  # length of lattice
    t_max = params.t_max
    dt = 0.00025
    steps = int(np.ceil(t_max / dt))
    # S = 10  # number of simulations to average over
    replicas = params.replicas
    # =============================================================================
    # E_bind0 = np.log(kd_0/ka0)
    # E_bind1 = np.log(kd_1/ka1)
    # E_bind2 = np.log(kd_2/ka2)
    # =============================================================================

    # =============================================================================
    # #extras
    # =============================================================================

    Curve01 = (1 / (k_0 + k_1)) * (k_0 * Curve0 + k_1 * Curve1)
    Curve02 = (1 / (k_2 + k_0)) * (k_2 * Curve2 + k_0 * Curve0)
    Curve12 = (1 / (k_2 + k_1)) * (k_2 * Curve2 + k_1 * Curve1)
    Curve012 = (1 / (k_0 + k_1 + k_2)) * (k_0 * Curve0 + k_1 * Curve1 + k_2 * Curve2)

    pitch01 = (1 / (k_0 + k_1)) * (k_0 * pitch0 + k_1 * pitch1)
    pitch02 = (1 / (k_0 + k_2)) * (k_0 * pitch0 + k_2 * pitch2)
    pitch12 = (1 / (k_2 + k_1)) * (k_2 * pitch2 + k_1 * pitch1)
    pitch012 = (1 / (k_0 + k_1 + k_2)) * (k_0 * pitch0 + k_1 * pitch1 + k_2 * pitch2)

    # =============================================================================
    # #SIMULATION
    #
    # =============================================================================

    # initialising arrays

    Latticeline = np.zeros((replicas, steps, lattice_length))
    LocCurveT = np.zeros((replicas, steps, lattice_length))
    GlobPitchtot = np.zeros((steps, replicas))
    GlobCurvetot = np.zeros((steps, replicas))
    BendEnergyDiff = np.zeros((steps, replicas))

    EmemTot = np.zeros((steps, replicas))
    EbindTot = np.zeros((steps, replicas))
    EbendTot = np.zeros((steps, replicas))

    # Koff0time = np.zeros((S,Q))
    # Koff1time = np.zeros((S,Q))

    # =============================================================================
    # #Defining a function re-calculating curvature energies
    #
    # =============================================================================
    def kot_r2tau(ktot, r):
        return (1 / ktot) * np.log(1 / r) if (ktot and r) else np.inf  # strictly >0

    label_2_loc_curve_delta = (
        Curve0,
        Curve01,
        Curve1,
        Curve12,
        Curve2,
        Curve02,
        Curve012,
    )

    label_2_glob_curve_pitch = (
        (Curve0, pitch0),
        (Curve01, pitch01),
        (Curve1, pitch1),
        (Curve12, pitch12),
        (Curve2, pitch2),
        (Curve02, pitch02),
        (Curve012, pitch012),
    )

    def glob_no(curv, pitch):
        gc = (8 / ((curv * pitch) ** -2 + 1)) if (curv and pitch) else 0
        cp = curv * pitch
        if cp > 1:
            gc = cp - 1 + 4
        return np.pi * k_mem * gc

    def bending_take_get_lims(site, N_neigh):
        if site <= N_neigh:
            LimS = 0
            LimB = site + N_neigh
        elif site >= lattice_length - N_neigh:
            LimS = site - N_neigh
            LimB = lattice_length
        elif site > N_neigh and site < lattice_length - N_neigh:
            LimS = site - N_neigh
            LimB = site + N_neigh
        else:
            raise ValueError
        return LimS, LimB

    def bending_take(LocCurve, site, c, Lattice, CurveSub, CurveNoSub, arg):

        # global Ebend_takeaway # not used anywhere else
        Ebend_takeaway = 0
        LocCurveCalc = 0
        if arg == 1:
            if c == 0:
                LocCurveCalc = 0
            else:
                LocCurveCalc = 0 + ((1 / c) * CurveNoSub - (1 / c) * CurveSub)

        # =============================================================================
        #     elif arg == 0:
        #         Ebend_takeaway = 0
        #         LocCurveCalc = 0
        # =============================================================================

        LimS, LimB = bending_take_get_lims(site, N_neigh)

        LocCurveAbs = 0

        for i in range(lattice_length):

            if i <= LimB and i >= LimS:
                LocCurveAbs = float(LocCurve[i]) + LocCurveCalc
            else:
                LocCurveAbs = LocCurve[i]

            if Lattice[i] == 0:
                Ebend_takeaway += (1 / 2) * k_0 * (LocCurveAbs - Curve0) ** 2
            elif Lattice[i] == 1:
                Ebend_takeaway += (1 / 2) * k_0 * (LocCurveAbs - Curve0) ** 2 + (
                    1 / 2
                ) * k_1 * (LocCurveAbs - Curve1) ** 2
            elif Lattice[i] == 2:
                Ebend_takeaway += (1 / 2) * k_1 * (LocCurveAbs - Curve1) ** 2
            elif Lattice[i] == 3:
                Ebend_takeaway += (1 / 2) * k_2 * (LocCurveAbs - Curve2) ** 2 + (
                    1 / 2
                ) * k_1 * (LocCurveAbs - Curve1) ** 2
            elif Lattice[i] == 4:
                Ebend_takeaway += (1 / 2) * k_2 * (LocCurveAbs - Curve2) ** 2
            elif Lattice[i] == 5:
                Ebend_takeaway += (1 / 2) * k_0 * (LocCurveAbs - Curve0) ** 2 + (
                    1 / 2
                ) * k_2 * (LocCurveAbs - Curve2) ** 2
            elif Lattice[i] == 6:
                Ebend_takeaway += (
                    (1 / 2) * k_0 * (LocCurveAbs - Curve0) ** 2
                    + (1 / 2) * k_1 * (LocCurveAbs - Curve1) ** 2
                    + (1 / 2) * k_2 * (LocCurveAbs - Curve2) ** 2
                )

        if arg == 1:
            Ebend_takeaway = (
                Ebend_takeaway - (1 / 2) * k_0 * (LocCurveAbs - CurveSub) ** 2
            )

        return Ebend_takeaway

    def get_lims(site, N_neigh):
        if site <= N_neigh:
            J = site
            LimS = -J
            LimB = N_neigh
        elif site >= lattice_length - N_neigh:
            J = lattice_length - site
            LimS = -N_neigh
            LimB = J
        elif site > N_neigh and site < lattice_length - N_neigh:
            LimS = -N_neigh
            LimB = N_neigh
        else:
            raise ValueError
        return LimS, LimB

    for replica in range(replicas):

        # initialize
        Emem = 0.0
        Lattice = []
        LocCurve = []
        for i in range(lattice_length):

            Lattice.append(7)  # all start in state 7-> no subunits
            LocCurve.append(0.0)

        # starting the simulation

        for step in range(steps):
            g: int = 0
            GlobCurve = 0.0
            GlobPitch = 0.0
            for site in range(lattice_length):
                # site = N-j-1
                r1 = random()
                r2 = random()
                r3 = random()

                Latticeline[replica][step][site] = Lattice[site]
                LocCurveT[replica][step][site] = LocCurve[site]
                LocCurve[site] = 0

                # =============================================================================
                #         #calculating the average local curvature of the site depending on its N_neigh neighbours
                #
                # =============================================================================

                c = 0

                for n in range(*get_lims(site, N_neigh)):
                    site_n = site + n
                    lat_site_n = Lattice[site_n]
                    if lat_site_n < 7:
                        c += 1
                        LocCurve[site] += label_2_loc_curve_delta[lat_site_n]

                if c != 0:
                    LocCurve[site] /= c

                if site == 0:
                    for site2 in range(lattice_length):
                        lat_site_n = Lattice[site2]
                        if lat_site_n < 7:
                            g += 1
                            GlobCurve += label_2_glob_curve_pitch[lat_site_n][0]
                            GlobPitch += label_2_glob_curve_pitch[lat_site_n][1]

                    GlobPitch = GlobPitch / lattice_length
                    if g != 0:
                        GlobCurve /= g
                # =============================================================================
                # #looping through sites to calculate on and off rates and changing site accordingly
                #
                # =============================================================================

                if Lattice[site] == 0:  # subunit 0 only
                    E_bend0 = (1/2)*k_0*(LocCurve[site]-Curve0)**2
                    if E_bend0 > Th:
                        Vps4_0 = E_v
                    else:
                        Vps4_0 = 0
                        
                    EbendTot[step][replica] = bending_take(
                        LocCurve, site, c, Lattice, Curve0, 0, 0
                    )
                    EbendNo0 = bending_take(LocCurve, site, c, Lattice, Curve0, 0, 1)
                    Ebind0 = BindMem0
                    Emem = glob_no(GlobCurve, GlobPitch)

                    GlobCurve_no0 = GlobCurve - Curve0 / g
                    GlobPitch_no0 = GlobPitch - pitch0 / lattice_length
                    GlobCurve_ad1 = GlobCurve - Curve0 / g + Curve01 / g
                    GlobPitch_ad1 = (
                        GlobPitch - pitch0 / lattice_length + pitch01 / lattice_length
                    )
                    GlobCurve_ad2 = GlobCurve - Curve0 / g + Curve02 / g
                    GlobPitch_ad2 = (
                        GlobPitch - pitch0 / lattice_length + pitch02 / lattice_length
                    )

                    Emem_ad1 = glob_no(GlobCurve_ad1, GlobPitch_ad1)

                    Emem_ad2 = glob_no(GlobCurve_ad2, GlobPitch_ad2)

                    Emem_no0 = glob_no(GlobCurve_no0, GlobPitch_no0)

                    Koff0 = (
                        np.exp(Ebind0)
                        * np.exp(EbendTot[step][replica] - EbendNo0)
                        * np.exp(Emem - Emem_no0)*np.exp(Vps4_0)
                    )

                    BendEnergyDiff[step][replica] = EbendTot[step][replica] - EbendNo0

                    Ktot = Kon1 + Kon2 + Koff0
                    tau = kot_r2tau(Ktot, r1)
                    Num_stay = (
                        tau / dt
                    )  # how many dts it should take for the state to change
                    if r3 < 1 / Num_stay:
                        if r2 < Kon1 / Ktot:
                            GlobCurve = GlobCurve_ad1
                            GlobPitch = GlobPitch_ad1
                            Emem = Emem_ad1
                            Lattice[site] = 1
                        elif r2 > (Kon1 + Koff0) / Ktot:
                            GlobCurve = GlobCurve_ad2
                            GlobPitch = GlobPitch_ad2
                            Emem = Emem_ad2
                            Lattice[site] = 5
                        else:
                            GlobCurve = GlobCurve_no0
                            GlobPitch = GlobPitch_no0
                            g -= 1
                            Emem = Emem_no0
                            Lattice[site] = 7

                elif Lattice[site] == 1:
                    E_bend0 = (1/2)*k_0*(LocCurve[site]-Curve0)**2
                    E_bend1 = (1/2)*k_1*(LocCurve[site]-Curve1)**2
                    if E_bend0 > Th:
                        Vps4_0 = E_v
                    else:
                        Vps4_0 = 0
                    if E_bend1 > Th:
                        Vps4_1 = E_v
                    else:
                        Vps4_1 = 0
                    
                    EbendNo0 = bending_take(
                        LocCurve, site, c, Lattice, Curve0, Curve1, 1
                    )
                    EbendNo1 = bending_take(
                        LocCurve, site, c, Lattice, Curve1, Curve0, 1
                    )
                    Ebind0 = BindFil0 + BindMem0
                    Ebind1 = BindFil1 + BindMem1
                    EbendTot[step][replica] = bending_take(
                        LocCurve, site, c, Lattice, Curve1, Curve0, 0
                    )
                    Emem = glob_no(GlobCurve, GlobPitch)

                    GlobCurve_no0 = GlobCurve + Curve1 / g - Curve01 / g
                    GlobPitch_no0 = (
                        GlobPitch + pitch1 / lattice_length - pitch01 / lattice_length
                    )
                    GlobCurve_no1 = GlobCurve + Curve0 / g - Curve01 / g
                    GlobPitch_no1 = (
                        GlobPitch + pitch0 / lattice_length - pitch01 / lattice_length
                    )
                    GlobCurve_ad2 = GlobCurve - Curve01 / g + Curve012 / g
                    GlobPitch_ad2 = (
                        GlobPitch - pitch01 / lattice_length + pitch012 / lattice_length
                    )

                    Emem_no0 = glob_no(GlobCurve_no0, GlobPitch_no0)

                    Emem_no1 = glob_no(GlobCurve_no1, GlobPitch_no1)

                    Emem_ad2 = glob_no(GlobCurve_ad2, GlobPitch_ad2)

                    Koff0 = (
                        np.exp(Ebind0)
                        * np.exp(EbendTot[step][replica] - EbendNo0)
                        * np.exp(Emem - Emem_no0)*np.exp(Vps4_0)
                    )
                    Koff1 = (
                        np.exp(Ebind1)
                        * np.exp(EbendTot[step][replica] - EbendNo1)
                        * np.exp(Emem - Emem_no1)*np.exp(Vps4_1)
                    )

                    BendEnergyDiff[step][replica] = (
                        EbendTot[step][replica]
                        - EbendNo0
                        + EbendTot[step][replica]
                        - EbendNo1
                    )

                    Ktot = Koff0 + Kon2 + Koff1
                    tau = kot_r2tau(Ktot, r1)
                    Num_stay = tau / dt
                    if r3 < 1 / Num_stay:
                        if r2 < Kon2 / Ktot:
                            GlobCurve = GlobCurve_ad2
                            GlobPitch = GlobPitch_ad2
                            Emem = Emem_ad2
                            Lattice[site] = 6
                        elif r2 > (Kon2 + Koff0) / Ktot:
                            GlobCurve = GlobCurve_no1
                            GlobPitch = GlobPitch_no1
                            Emem = Emem_no1
                            Lattice[site] = 0
                        else:
                            GlobCurve = GlobCurve_no0
                            GlobPitch = GlobPitch_no0
                            Emem = Emem_no0
                            Lattice[site] = 2

                elif Lattice[site] == 2:
                    Ebind1 = BindMem1
                    # EbindTot[q][s] += E_bind1
                    EbendNo1 = bending_take(LocCurve, site, c, Lattice, Curve1, 0, 1)
                    EbendTot[step][replica] = bending_take(
                        LocCurve, site, c, Lattice, Curve1, 0, 0
                    )
                    E_bend1 = (1/2)*k_1*(LocCurve[site]-Curve1)**2

                    if E_bend1 > Th:
                        Vps4_1 = E_v
                    else:
                        Vps4_1 = 0

                    Emem = glob_no(GlobCurve, GlobPitch)

                    GlobCurve_no1 = GlobCurve - Curve1 / (g)
                    GlobPitch_no1 = GlobPitch - pitch1 / lattice_length
                    GlobCurve_ad0 = GlobCurve - Curve1 / (g) + Curve01 / g
                    GlobPitch_ad0 = (
                        GlobPitch - pitch1 / lattice_length + pitch01 / lattice_length
                    )
                    GlobCurve_ad2 = GlobCurve - Curve1 / (g) + Curve12 / g
                    GlobPitch_ad2 = (
                        GlobPitch - pitch1 / lattice_length + pitch12 / lattice_length
                    )

                    Emem_ad0 = glob_no(GlobCurve_ad0, GlobPitch_ad0)

                    Emem_ad2 = glob_no(GlobCurve_ad2, GlobPitch_ad2)

                    Emem_no1 = glob_no(GlobCurve_no1, GlobPitch_no1)

                    Koff1 = (
                        np.exp(Ebind1)
                        * np.exp(EbendTot[step][replica] - EbendNo1)
                        * np.exp(Emem - Emem_no1)*np.exp(Vps4_1)
                    )

                    BendEnergyDiff[step][replica] = EbendTot[step][replica] - EbendNo1

                    Ktot = Kon0 + Kon2 + Koff1
                    tau = kot_r2tau(Ktot, r1)
                    Num_stay = tau / dt
                    if r3 < 1 / Num_stay:
                        if r2 < Kon0 / Ktot:
                            GlobCurve = GlobCurve_ad0
                            GlobPitch = GlobPitch_ad0
                            Emem = Emem_ad0
                            Lattice[site] = 1
                        elif r2 > (Kon0 + Kon2) / Ktot:
                            GlobCurve = GlobCurve_no1
                            GlobPitch = GlobPitch_no1
                            Emem = Emem_no1
                            g -= 1
                            Lattice[site] = 7
                        else:
                            GlobCurve = GlobCurve_ad2
                            GlobPitch = GlobPitch_ad2
                            Emem = Emem_ad2
                            Lattice[site] = 3

                elif Lattice[site] == 3:
                    
                    E_bend2 = (1/2)*k_2*(LocCurve[site]-Curve2)**2
                    E_bend1 = (1/2)*k_1*(LocCurve[site]-Curve1)**2
                    if E_bend1 > Th:
                        Vps4_1 = E_v
                    else:
                        Vps4_1 = 0
                    if E_bend2 > Th:
                        Vps4_2 = E_v
                    else:
                        Vps4_2 = 0
                    
                    Ebind1 = BindMem1 + BindFil1
                    Ebind2 = BindMem2 + BindFil2
                    EbendNo1 = bending_take(
                        LocCurve, site, c, Lattice, Curve1, Curve2, 1
                    )
                    EbendNo2 = bending_take(
                        LocCurve, site, c, Lattice, Curve2, Curve1, 1
                    )
                    EbendTot[step][replica] = bending_take(
                        LocCurve, site, c, Lattice, Curve2, Curve1, 0
                    )

                    Emem = glob_no(GlobCurve, GlobPitch)

                    GlobCurve_no2 = GlobCurve + Curve1 / g - Curve12 / g
                    GlobPitch_no2 = (
                        GlobPitch + pitch1 / lattice_length - pitch12 / lattice_length
                    )
                    GlobCurve_no1 = GlobCurve + Curve2 / g - Curve12 / g
                    GlobPitch_no1 = (
                        GlobPitch + pitch2 / lattice_length - pitch12 / lattice_length
                    )
                    GlobCurve_ad0 = GlobCurve - Curve12 / g + Curve012 / g
                    GlobPitch_ad0 = (
                        GlobPitch - pitch12 / lattice_length + pitch012 / lattice_length
                    )

                    Emem_ad0 = glob_no(GlobCurve_ad0, GlobPitch_ad0)

                    Emem_no1 = glob_no(GlobCurve_no1, GlobPitch_no1)

                    Emem_no2 = glob_no(GlobCurve_no2, GlobPitch_no2)

                    Koff2 = (
                        np.exp(Ebind2)
                        * np.exp(EbendTot[step][replica] - EbendNo2)
                        * np.exp(Emem - Emem_no2)*np.exp(Vps4_2)
                    )
                    Koff1 = (
                        np.exp(Ebind1)
                        * np.exp(EbendTot[step][replica] - EbendNo1)
                        * np.exp(Emem - Emem_no1)*np.exp(Vps4_1)
                    )
                    BendEnergyDiff[step][replica] = (
                        EbendTot[step][replica]
                        - EbendNo2
                        + EbendTot[step][replica]
                        - EbendNo1
                    )

                    Ktot = Kon0 + Koff2 + Koff1
                    tau = kot_r2tau(Ktot, r1)
                    Num_stay = tau / dt
                    if r3 < 1 / Num_stay:
                        if r2 < Kon0 / Ktot:
                            GlobCurve = GlobCurve_ad0
                            GlobPitch = GlobPitch_ad0
                            Emem = Emem_ad0
                            Lattice[site] = 6
                        elif r2 > (Kon0 + Koff1) / Ktot:
                            GlobCurve = GlobCurve_no2
                            GlobPitch = GlobPitch_no2
                            Emem = Emem_no2
                            Lattice[site] = 2
                        else:
                            GlobCurve = GlobCurve_no1
                            GlobPitch = GlobPitch_no1
                            Emem = Emem_no1
                            Lattice[site] = 4

                elif Lattice[site] == 4:
                    E_bend2 =(1/2)*k_2*(LocCurve[site]-Curve2)**2

                    if E_bend2 > Th:
                        Vps4_2 = E_v
                    else:
                        Vps4_2 = 0
                    Ebind2 = BindMem2

                    Emem = glob_no(GlobCurve, GlobPitch)

                    EbendTot[step][replica] = bending_take(
                        LocCurve, site, c, Lattice, Curve2, 0, 0
                    )
                    EbendNo2 = bending_take(LocCurve, site, c, Lattice, Curve2, 0, 1)

                    GlobCurve_no2 = GlobCurve - Curve2 / g
                    GlobPitch_no2 = GlobPitch - pitch2 / lattice_length
                    GlobCurve_ad1 = GlobCurve - Curve2 / g + Curve12 / g
                    GlobPitch_ad1 = (
                        GlobPitch - pitch2 / lattice_length + pitch12 / lattice_length
                    )
                    GlobCurve_ad0 = GlobCurve - Curve2 / g + Curve02 / g
                    GlobPitch_ad0 = (
                        GlobPitch - pitch2 / lattice_length + pitch02 / lattice_length
                    )

                    Emem_ad0 = glob_no(GlobCurve_ad0, GlobPitch_ad0)

                    Emem_ad1 = glob_no(GlobCurve_ad1, GlobPitch_ad1)

                    Emem_no2 = glob_no(GlobCurve_no2, GlobPitch_no2)

                    Koff2 = (
                        np.exp(Ebind2)
                        * np.exp(EbendTot[step][replica] - EbendNo2)
                        * np.exp(Emem - Emem_no2)*np.exp(Vps4_2)   
                    )
                    BendEnergyDiff[step][replica] = EbendTot[step][replica] - EbendNo2

                    Ktot = Kon0 + Kon1 + Koff2
                    tau = kot_r2tau(Ktot, r1)
                    Num_stay = tau / dt
                    if r3 < 1 / Num_stay:
                        if r2 < Kon1 / Ktot:
                            GlobCurve = GlobCurve_ad1
                            GlobPitch = GlobPitch_ad1
                            Emem = Emem_ad1
                            Lattice[site] = 3
                        elif r2 > (Kon0 + Kon1) / Ktot:
                            GlobCurve = GlobCurve_no2
                            GlobPitch = GlobPitch_no2
                            Emem = Emem_no2
                            g -= 1
                            Lattice[site] = 7
                        else:
                            GlobCurve = GlobCurve_ad0
                            GlobPitch = GlobPitch_ad0
                            Emem = Emem_ad0
                            Lattice[site] = 5

                elif Lattice[site] == 5:
                    E_bend0 = (1/2)*k_0*(LocCurve[site]-Curve0)**2
                    E_bend2 = (1/2)*k_2*(LocCurve[site]-Curve2)**2
                    if E_bend0 > Th:
                        Vps4_0 = E_v
                    else:
                        Vps4_0 = 0
                    if E_bend2 > Th:
                        Vps4_2 = E_v
                    else:
                        Vps4_2 = 0
                        
                    Ebind2 = BindMem2 + BindFil2
                    Ebind0 = BindMem0 + BindFil0
                    EbendTot[step][replica] = bending_take(
                        LocCurve, site, c, Lattice, Curve0, Curve2, 0
                    )

                    EbendNo0 = bending_take(
                        LocCurve, site, c, Lattice, Curve0, Curve2, 1
                    )
                    EbendNo2 = bending_take(
                        LocCurve, site, c, Lattice, Curve2, Curve0, 1
                    )

                    Emem = glob_no(GlobCurve, GlobPitch)

                    GlobCurve_no0 = GlobCurve + Curve2 / g - Curve02 / g
                    GlobPitch_no0 = (
                        GlobPitch + pitch2 / lattice_length - pitch02 / lattice_length
                    )
                    GlobCurve_no2 = GlobCurve + Curve0 / g - Curve02 / g
                    GlobPitch_no2 = (
                        GlobPitch + pitch0 / lattice_length - pitch02 / lattice_length
                    )
                    GlobCurve_ad1 = GlobCurve - Curve02 / g + Curve012 / g
                    GlobPitch_ad1 = (
                        GlobPitch - pitch02 / lattice_length + pitch012 / lattice_length
                    )

                    Emem_ad1 = glob_no(GlobCurve_ad1, GlobPitch_ad1)

                    Emem_no0 = glob_no(GlobCurve_no0, GlobPitch_no0)

                    Emem_no2 = glob_no(GlobCurve_no2, GlobPitch_no2)

                    Koff0 = (
                        np.exp(Ebind0)
                        * np.exp(EbendTot[step][replica] - EbendNo0)
                        * np.exp(Emem - Emem_no0)*np.exp(Vps4_0)  
                    )
                    Koff2 = (
                        np.exp(Ebind2)
                        * np.exp(EbendTot[step][replica] - EbendNo2)
                        * np.exp(Emem - Emem_no2)*np.exp(Vps4_2)  
                    )
                    BendEnergyDiff[step][replica] = (
                        EbendTot[step][replica]
                        - EbendNo2
                        + EbendTot[step][replica]
                        - EbendNo0
                    )

                    Ktot = Koff0 + Koff2 + Kon1
                    tau = kot_r2tau(Ktot, r1)
                    Num_stay = tau / dt
                    if r3 < 1 / Num_stay:
                        if r2 < Kon1 / Ktot:
                            GlobCurve = GlobCurve_ad1
                            GlobPitch = GlobPitch_ad1
                            Emem = Emem_ad1
                            Lattice[site] = 6
                        elif r2 > (Kon1 + Koff0) / Ktot:
                            GlobCurve = GlobCurve_no2
                            GlobPitch = GlobPitch_no2
                            Emem = Emem_no2
                            Lattice[site] = 0
                        else:
                            GlobCurve = GlobCurve_no0
                            GlobPitch = GlobPitch_no0
                            Emem = Emem_no0
                            Lattice[site] = 4

                elif Lattice[site] == 6:
                    Ebind2 = BindMem2 + BindFil2
                    Ebind0 = BindMem0 + BindFil0
                    Ebind1 = BindMem1 + BindFil1
                    
                    E_bend0 = (1/2)*k_0*(LocCurve[site]-Curve0)**2
                    E_bend1 = (1/2)*k_1*(LocCurve[site]-Curve1)**2
                    E_bend2 = (1/2)*k_2*(LocCurve[site]-Curve2)**2
                    if E_bend0 > Th:
                        Vps4_0 = E_v
                    else:
                        Vps4_0 = 0
                    if E_bend1 > Th:
                        Vps4_1 = E_v
                    else:
                        Vps4_1 = 0
                    if E_bend2 > Th:
                        Vps4_2 = E_v
                    else:
                        Vps4_2 = 0

                    EbendTot[step][replica] = bending_take(
                        LocCurve, site, c, Lattice, Curve0, Curve12, 0
                    )

                    EbendNo0 = bending_take(
                        LocCurve, site, c, Lattice, Curve0, Curve12, 1
                    )
                    EbendNo1 = bending_take(
                        LocCurve, site, c, Lattice, Curve1, Curve02, 1
                    )
                    EbendNo2 = bending_take(
                        LocCurve, site, c, Lattice, Curve2, Curve01, 1
                    )

                    # emem
                    Emem = glob_no(GlobCurve, GlobPitch)

                    GlobCurve_no0 = GlobCurve + Curve12 / g - Curve012 / g
                    GlobPitch_no0 = (
                        GlobPitch + pitch12 / lattice_length - pitch012 / lattice_length
                    )
                    GlobCurve_no2 = GlobCurve + Curve01 / g - Curve012 / g
                    GlobPitch_no2 = (
                        GlobPitch + pitch01 / lattice_length - pitch012 / lattice_length
                    )
                    GlobCurve_no1 = GlobCurve + Curve02 / g - Curve012 / g
                    GlobPitch_no1 = (
                        GlobPitch + pitch02 / lattice_length - pitch012 / lattice_length
                    )

                    # no0,no1,no2

                    Emem_no0 = glob_no(GlobCurve_no0, GlobPitch_no0)

                    Emem_no1 = glob_no(GlobCurve_no1, GlobPitch_no1)

                    Emem_no2 = glob_no(GlobCurve_no2, GlobPitch_no2)

                    Koff0 = (
                        np.exp(Ebind0)
                        * np.exp(EbendTot[step][replica] - EbendNo0)
                        * np.exp(Emem - Emem_no0)*np.exp(Vps4_0)  
                    )
                    Koff2 = (
                        np.exp(Ebind2)
                        * np.exp(EbendTot[step][replica] - EbendNo2)
                        * np.exp(Emem - Emem_no2)*np.exp(Vps4_2)  
                    )
                    Koff1 = (
                        np.exp(Ebind1)
                        * np.exp(EbendTot[step][replica] - EbendNo1)
                        * np.exp(Emem - Emem_no1)*np.exp(Vps4_1)  
                    )
                    BendEnergyDiff[step][replica] = (
                        EbendTot[step][replica]
                        - EbendNo2
                        + EbendTot[step][replica]
                        - EbendNo1
                        + EbendTot[step][replica]
                        - EbendNo0
                    )

                    Ktot = Koff0 + Koff2 + Koff1
                    tau = kot_r2tau(Ktot, r1)
                    Num_stay = tau / dt
                    if r3 < 1 / Num_stay:
                        if r2 < Koff0 / Ktot:
                            GlobCurve = GlobCurve_no0
                            GlobPitch = GlobPitch_no0
                            Emem = Emem_no0
                            Lattice[site] = 3
                        elif r2 > (Koff0 + Koff1) / Ktot:
                            GlobCurve = GlobCurve_no2
                            GlobPitch = GlobPitch_no2
                            Emem = Emem_no2
                            Lattice[site] = 1
                        else:
                            GlobCurve = GlobCurve_no1
                            GlobPitch = GlobPitch_no1
                            Emem = Emem_no1
                            Lattice[site] = 5
                else:
                    EbendTot[step][replica] = bending_take(
                        LocCurve, site, c, Lattice, 0, 0, 0
                    )

                    Emem = glob_no(GlobCurve, GlobPitch)

                    GlobCurve_ad0 = GlobCurve + Curve0 / (g + 1)
                    GlobPitch_ad0 = GlobPitch + pitch0 / lattice_length
                    GlobCurve_ad2 = GlobCurve + Curve2 / (g + 1)
                    GlobPitch_ad2 = GlobPitch + pitch2 / lattice_length
                    GlobCurve_ad1 = GlobCurve + Curve1 / (g + 1)
                    GlobPitch_ad1 = GlobPitch + pitch1 / lattice_length

                    # ad0,ad1,ad2

                    Emem_ad0 = glob_no(GlobCurve_ad0, GlobPitch_ad0)

                    Emem_ad1 = glob_no(GlobCurve_ad1, GlobPitch_ad1)

                    Emem_ad2 = glob_no(GlobCurve_ad2, GlobPitch_ad2)

                    Ktot = Kon0 + Kon2 + Kon1
                    tau = kot_r2tau(Ktot, r1)
                    Num_stay = tau / dt
                    if r3 < 1 / Num_stay:
                        if r2 < Kon0 / Ktot:
                            GlobCurve = GlobCurve_ad0
                            GlobPitch = GlobPitch_ad0
                            Emem = Emem_ad0
                            g += 1
                            Lattice[site] = 0
                        elif r2 > (Kon0 + Kon1) / Ktot:
                            GlobCurve = GlobCurve_ad2
                            GlobPitch = GlobPitch_ad2
                            Emem = Emem_ad2
                            g += 1
                            Lattice[site] = 4
                        else:
                            GlobCurve = GlobCurve_ad1
                            GlobPitch = GlobPitch_ad1
                            Emem = Emem_ad1
                            g += 1
                            Lattice[site] = 2

            GlobPitchtot[step][replica] += GlobPitch
            GlobCurvetot[step][replica] += GlobCurve
            EmemTot[step][replica] += Emem

    return Res(
        out=(
            Latticeline,
            LocCurveT,
            GlobPitchtot,
            GlobCurvetot,
            BendEnergyDiff,
            EmemTot,
            EbindTot,
            EbendTot,
        ),
        steps=steps,
        random_c=random_c,
    )


simulate_numba = numba.njit(simulate)


def main(params: Params):

    res = simulate(params)

    summary = summarize(params, res)

    return summary


main_numba = numba.njit(main, cache=True)
