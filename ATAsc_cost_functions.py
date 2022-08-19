# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 08:55:32 2022

@author: MGR

ATA-sc cost functions


Parameters:
- ab: Parameters a2, a3, b2 (to fit)
- x2_IC: initial condition for x2. (One number or vector) 
- tspan: time at which the samples were taken
- data_x2: x2 data to fit - one vector per timepoint in tspan

- wd_param (Wasserstein distance parameters):
    - 
    - 
    - 
    
    
- scaling_factors: factors to scale parameters
- noise_mat: Already initialized noise matrix

- x1_expression_interpolated


WD parameters:
- stdev(1), Ntrunc(40), step(0.1)

"""

import numpy as np

from wasserstein_distance import wasserstein_distance_1D
from modelClass_ODE_SDE_x2 import runEM_initNoise_x1int_x2ICs, runEM_initNoise_x2ICs
from modelClass_ODE_SDE_x2 import runEM_initNoise_x1int_x2ICs_ncell


###############################################################################
def sde_x2_cf_Scaled_InitNoise_x1int_x2ICs_l1(ab, x2_IC, tspan, data_x2, wd_param, scaling_factors, noise_mat, x1_expression_interpolated):
    # print(ab)
    ab_sc = np.array(ab)/scaling_factors
    # print(ab_sc)
    modelParameters = {'a2':ab_sc[0], 'a3': ab_sc[1], 'b2': ab_sc[2]}
    timeParameters = {'t_start':0, 't_stop':tspan[-1], 'steps':tspan[-1]}
    
    _, result_SDE_it = runEM_initNoise_x1int_x2ICs(timeParameters, x2_IC, modelParameters, noise_mat, x1_expression_interpolated)
    # Dstack (time, x, cell)
    result_SDE_it = np.dstack(result_SDE_it)
    
    wd = 0
    for time in tspan:
        real = data_x2[str(time)]
        simulated = np.transpose(result_SDE_it[:,time,:])
        wd_x2 = wasserstein_distance_1D(real, simulated, wd_param[0], wd_param[1], wd_param[2])
        wd += wd_x2
    
    #print(wd)
    return wd


###############################################################################
def sde_x2_cf_Scaled_InitNoise_x1int_x2ICs_l2(ab, x2_IC, tspan, data_x2, wd_param, scaling_factors, noise_mat, x1_expression_interpolated):
    # print(ab)
    ab_sc = np.array(ab)/scaling_factors
    # print(ab_sc)
    modelParameters = {'a2':ab_sc[0], 'a3': ab_sc[1], 'b2': ab_sc[2]}
    timeParameters = {'t_start':0, 't_stop':tspan[-1], 'steps':tspan[-1]}
    
    _, result_SDE_it = runEM_initNoise_x1int_x2ICs(timeParameters, x2_IC, modelParameters, noise_mat, x1_expression_interpolated)
    # Dstack (time, x, cell)
    result_SDE_it = np.dstack(result_SDE_it)
    
    wd = 0
    time0 = tspan[0]
    for time in tspan:
        real = data_x2[str(time)]
        simulated = np.transpose(result_SDE_it[:,time-time0,:])
        #simulated = np.transpose(result_SDE_it[:,time,:])
        wd_x2 = wasserstein_distance_1D(real, simulated, wd_param[0], wd_param[1], wd_param[2])
        wd += wd_x2**2
    
    wd = np.sqrt(wd)
    #print(wd)
    return wd


###############################################################################
def sde_x2_cf_Scaled_InitNoise_alone_x2ICs_l1(ab, x2_IC, tspan, data_x2, wd_param, scaling_factors, noise_mat):
    # print(ab)
    ab_sc = np.array(ab)/scaling_factors
    # print(ab_sc)
    modelParameters = {'a2':ab_sc[0], 'a3': 0, 'b2': ab_sc[1]}
    timeParameters = {'t_start':0, 't_stop':tspan[-1], 'steps':tspan[-1]}
    
    _, result_SDE_it = runEM_initNoise_x2ICs(timeParameters, x2_IC, modelParameters, noise_mat)
    # Dstack (time, x, cell)
    result_SDE_it = np.dstack(result_SDE_it)
    
    wd = 0
    for time in tspan:
        real = data_x2[str(time)]
        simulated = np.transpose(result_SDE_it[:,time,:])
        wd_x2 = wasserstein_distance_1D(real, simulated, wd_param[0], wd_param[1], wd_param[2])
        wd += wd_x2
    
    #print(wd)
    return wd

###############################################################################
def sde_x2_cf_Scaled_InitNoise_alone_x2ICs_l2(ab, x2_IC, tspan, data_x2, wd_param, scaling_factors, noise_mat):
    # print(ab)
    ab_sc = np.array(ab)/scaling_factors
    # print(ab_sc)
    modelParameters = {'a2':ab_sc[0], 'a3': 0, 'b2': ab_sc[1]}
    timeParameters = {'t_start':0, 't_stop':tspan[-1], 'steps':tspan[-1]}
    
    _, result_SDE_it = runEM_initNoise_x2ICs(timeParameters, x2_IC, modelParameters, noise_mat)
    # Dstack (time, x, cell)
    result_SDE_it = np.dstack(result_SDE_it)
    
    wd = 0
    time0 = tspan[0]
    for time in tspan:
        real = data_x2[str(time)]
        simulated = np.transpose(result_SDE_it[:,time-time0,:])
        #simulated = np.transpose(result_SDE_it[:,time,:])
        wd_x2 = wasserstein_distance_1D(real, simulated, wd_param[0], wd_param[1], wd_param[2])
        wd += wd_x2**2
    
    wd = np.sqrt(wd)
    #print(wd)
    return wd





###############################################################################
def sde_x2_cf_Scaled_InitNoise_x1int_x2ICs_ncell_l2(ab, x2_IC, tspan, data_x2, wd_param, scaling_factors, noise_mat, x1_expression_interpolated, cells_id):
    # print(ab)
    ab_sc = np.array(ab)/scaling_factors
    # print(ab_sc)
    modelParameters = {'a2':ab_sc[0], 'a3': ab_sc[1], 'b2': ab_sc[2]}
    timeParameters = {'t_start':0, 't_stop':tspan[-1], 'steps':tspan[-1]}
    
    
    _, result_SDE_it = runEM_initNoise_x1int_x2ICs_ncell(timeParameters, x2_IC, modelParameters, noise_mat, x1_expression_interpolated, cells_id)
    # Dstack (time, x, cell)
    result_SDE_it = np.dstack(result_SDE_it)
    
    wd = 0
    time0 = tspan[0]
    for time in tspan:
        real = data_x2[str(time)]
        simulated = np.transpose(result_SDE_it[:,time-time0,:])
        wd_x2 = wasserstein_distance_1D(real, simulated, wd_param[0], wd_param[1], wd_param[2])
        wd += wd_x2**2
    
    wd = np.sqrt(wd)
    #print(wd)
    return wd