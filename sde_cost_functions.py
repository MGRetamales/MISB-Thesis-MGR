# -*- coding: utf-8 -*-
"""
Created on Fri May  6 08:45:53 2022

@author: MGR


Possible objective functions to try in estimating parameters
- Scaled/not Scaled
- Init noise / not init Noise
- L1 / L2 norms 


- sde_cf_notScaled_notInitNoise_l1 
- sde_cf_notScaled_notInitNoise_l2
- sde_cf_Scaled_notInitNoise_l1 
- sde_cf_Scaled_notInitNoise_l2
- sde_cf_notScaled_InitNoise_l1 
- sde_cf_notScaled_InitNoise_l2
- sde_cf_Scaled_InitNoise_l1 
- sde_cf_Scaled_InitNoise_l2

Parameters:
- ab, x0, tspan, data, num_cells

- Aditional: scale, noise_mat


WD parameters:
- stdev(1), Ntrunc(40), step(0.1)

"""

import numpy as np

from wasserstein_distance import wasserstein_distance_1D
from modelClass_ODE_SDE import  runEulerMaruyama_noplot, runEulerMaruyama_initializedNoise
from modelClass_ODE_SDE_x2 import runEulerMaruyama_initializedNoise_x1interpolated


###############################################################################
def sde_cf_notScaled_notInitNoise_l1(ab, x0, tspan, data, num_cells, wd_param, scaling_factors, noise_mat):
    #print(ab)
    modelParameters = {'a1': ab[0], 'b1':ab[1], 'a2':ab[2], 'a3': ab[3], 'b2': ab[4]}
    timeParameters = {'t_start':0, 't_stop':tspan[-1], 'steps':tspan[-1]}
    
    _, result_SDE_it = runEulerMaruyama_noplot(timeParameters, num_cells, x0, modelParameters)
    # Dstack (time, x, cell)
    result_SDE_it = np.dstack(result_SDE_it)
    
    wd = 0
    for time in tspan:
        real = data[str(time)]
        simulated = np.transpose(result_SDE_it[time,:,:])
        wd_x1 = wasserstein_distance_1D(real[:,0], simulated[:,0], wd_param[0], wd_param[1], wd_param[2])
        wd_x2 = wasserstein_distance_1D(real[:,1], simulated[:,1], wd_param[0], wd_param[1], wd_param[2])
        wd += wd_x1+wd_x2
    print(wd)
    return wd

###############################################################################
def sde_cf_notScaled_notInitNoise_l2(ab, x0, tspan, data, num_cells, wd_param, scaling_factors, noise_mat):
    #print(ab)
    modelParameters = {'a1': ab[0], 'b1':ab[1], 'a2':ab[2], 'a3': ab[3], 'b2': ab[4]}
    timeParameters = {'t_start':0, 't_stop':tspan[-1], 'steps':tspan[-1]}
    
    _, result_SDE_it = runEulerMaruyama_noplot(timeParameters, num_cells, x0, modelParameters)
    # Dstack (time, x, cell)
    result_SDE_it = np.dstack(result_SDE_it)
    
    wd = 0
    for time in tspan:
        real = data[str(time)]
        simulated = np.transpose(result_SDE_it[time,:,:])
        wd_x1 = wasserstein_distance_1D(real[:,0], simulated[:,0], wd_param[0], wd_param[1], wd_param[2])
        wd_x2 = wasserstein_distance_1D(real[:,1], simulated[:,1], wd_param[0], wd_param[1], wd_param[2])
        wd += wd_x1**2+wd_x2**2
    
    wd = np.sqrt(wd)
    print(wd)
    return wd

###############################################################################
def sde_cf_Scaled_notInitNoise_l1(ab, x0, tspan, data, num_cells, wd_param, scaling_factors, noise_mat):
    
    ab_sc = np.array(ab)/scaling_factors
    #print(ab)
    #print(ab_sc)
    
    modelParameters = {'a1': ab_sc[0], 'b1':ab_sc[1], 'a2':ab_sc[2], 'a3': ab_sc[3], 'b2': ab_sc[4]}
    timeParameters = {'t_start':0, 't_stop':tspan[-1], 'steps':tspan[-1]}
    
    _, result_SDE_it = runEulerMaruyama_noplot(timeParameters, num_cells, x0, modelParameters)
    # Dstack (time, x, cell)
    result_SDE_it = np.dstack(result_SDE_it)
    
    wd = 0
    for time in tspan:
        real = data[str(time)]
        simulated = np.transpose(result_SDE_it[time,:,:])
        wd_x1 = wasserstein_distance_1D(real[:,0], simulated[:,0], wd_param[0], wd_param[1], wd_param[2])
        wd_x2 = wasserstein_distance_1D(real[:,1], simulated[:,1], wd_param[0], wd_param[1], wd_param[2])
        wd += wd_x1+wd_x2
        
    print(wd)
    return wd

###############################################################################
def sde_cf_Scaled_notInitNoise_l2(ab, x0, tspan, data, num_cells, wd_param, scaling_factors, noise_mat):
    
    ab_sc = np.array(ab)/scaling_factors
    #print(ab)
    #print(ab_sc)
    
    modelParameters = {'a1': ab_sc[0], 'b1':ab_sc[1], 'a2':ab_sc[2], 'a3': ab_sc[3], 'b2': ab_sc[4]}
    timeParameters = {'t_start':0, 't_stop':tspan[-1], 'steps':tspan[-1]}
    
    _, result_SDE_it = runEulerMaruyama_noplot(timeParameters, num_cells, x0, modelParameters)
    # Dstack (time, x, cell)
    result_SDE_it = np.dstack(result_SDE_it)
    
    wd = 0
    for time in tspan:
        real = data[str(time)]
        simulated = np.transpose(result_SDE_it[time,:,:])
        wd_x1 = wasserstein_distance_1D(real[:,0], simulated[:,0], wd_param[0], wd_param[1], wd_param[2])
        wd_x2 = wasserstein_distance_1D(real[:,1], simulated[:,1], wd_param[0], wd_param[1], wd_param[2])
        wd += wd_x1**2+wd_x2**2
        
    wd = np.sqrt(wd)
    print(wd)
    return wd


###############################################################################
def sde_cf_notScaled_InitNoise_l1(ab, x0, tspan, data, num_cells, wd_param, scaling_factors, noise_mat):
    # print(ab)
    modelParameters = {'a1': ab[0], 'b1':ab[1], 'a2':ab[2], 'a3': ab[3], 'b2': ab[4]}
    timeParameters = {'t_start':0, 't_stop':tspan[-1], 'steps':tspan[-1]}
    
    _, result_SDE_it = runEulerMaruyama_initializedNoise(timeParameters, num_cells, x0, modelParameters, noise_mat)
    # Dstack (time, x, cell)
    result_SDE_it = np.dstack(result_SDE_it)
    
    wd = 0
    for time in tspan:
        real = data[str(time)]
        simulated = np.transpose(result_SDE_it[time,:,:])
        wd_x1 = wasserstein_distance_1D(real[:,0], simulated[:,0], wd_param[0], wd_param[1], wd_param[2])
        wd_x2 = wasserstein_distance_1D(real[:,1], simulated[:,1], wd_param[0], wd_param[1], wd_param[2])
        wd += wd_x1+wd_x2
    
    print(wd)
    return wd

###############################################################################
def sde_cf_notScaled_InitNoise_l2(ab, x0, tspan, data, num_cells, wd_param, scaling_factors, noise_mat):
    # print(ab)
    modelParameters = {'a1': ab[0], 'b1':ab[1], 'a2':ab[2], 'a3': ab[3], 'b2': ab[4]}
    timeParameters = {'t_start':0, 't_stop':tspan[-1], 'steps':tspan[-1]}
    
    _, result_SDE_it = runEulerMaruyama_initializedNoise(timeParameters, num_cells, x0, modelParameters, noise_mat)
    # Dstack (time, x, cell)
    result_SDE_it = np.dstack(result_SDE_it)
    
    wd = 0
    for time in tspan:
        real = data[str(time)]
        simulated = np.transpose(result_SDE_it[time,:,:])
        wd_x1 = wasserstein_distance_1D(real[:,0], simulated[:,0], wd_param[0], wd_param[1], wd_param[2])
        wd_x2 = wasserstein_distance_1D(real[:,1], simulated[:,1], wd_param[0], wd_param[1], wd_param[2])
        wd += wd_x1**2+wd_x2**2
    
    wd = np.sqrt(wd)
    print(wd)
    return wd

###############################################################################
def sde_cf_Scaled_InitNoise_l1(ab, x0, tspan, data, num_cells, wd_param, scaling_factors, noise_mat):
    # print(ab)
    ab_sc = np.array(ab)/scaling_factors
    # print(ab_sc)
    modelParameters = {'a1': ab_sc[0], 'b1':ab_sc[1], 'a2':ab_sc[2], 'a3': ab_sc[3], 'b2': ab_sc[4]}
    timeParameters = {'t_start':0, 't_stop':tspan[-1], 'steps':tspan[-1]}
    
    _, result_SDE_it = runEulerMaruyama_initializedNoise(timeParameters, num_cells, x0, modelParameters, noise_mat)
    # Dstack (time, x, cell)
    result_SDE_it = np.dstack(result_SDE_it)
    
    wd = 0
    for time in tspan:
        real = data[str(time)]
        simulated = np.transpose(result_SDE_it[time,:,:])
        wd_x1 = wasserstein_distance_1D(real[:,0], simulated[:,0], wd_param[0], wd_param[1], wd_param[2])
        wd_x2 = wasserstein_distance_1D(real[:,1], simulated[:,1], wd_param[0], wd_param[1], wd_param[2])
        wd += wd_x1+wd_x2
    
    print(wd)
    return wd

###############################################################################
def sde_cf_Scaled_InitNoise_l2(ab, x0, tspan, data, num_cells, wd_param, scaling_factors, noise_mat):
    # print(ab)
    ab_sc = np.array(ab)/scaling_factors
    # print(ab_sc)
    modelParameters = {'a1': ab_sc[0], 'b1':ab_sc[1], 'a2':ab_sc[2], 'a3': ab_sc[3], 'b2': ab_sc[4]}
    timeParameters = {'t_start':0, 't_stop':tspan[-1], 'steps':tspan[-1]}
    
    _, result_SDE_it = runEulerMaruyama_initializedNoise(timeParameters, num_cells, x0, modelParameters, noise_mat)
    # Dstack (time, x, cell)
    result_SDE_it = np.dstack(result_SDE_it)
    
    wd = 0
    for time in tspan:
        real = data[str(time)]
        simulated = np.transpose(result_SDE_it[time,:,:])
        wd_x1 = wasserstein_distance_1D(real[:,0], simulated[:,0], wd_param[0], wd_param[1], wd_param[2])
        wd_x2 = wasserstein_distance_1D(real[:,1], simulated[:,1], wd_param[0], wd_param[1], wd_param[2])
        wd += wd_x1**2+wd_x2**2
    
    wd = np.sqrt(wd)
    print(wd)
    return wd



###############################################################################
# Cost Functions for only X2!! with x1 trajectories interpolated
###############################################################################


    

def sde_x2_cf_Scaled_InitNoise_l1(ab, x0, tspan, data_x2, num_cells, wd_param, scaling_factors, noise_mat, x1_expression_interpolated):
    # print(ab)
    ab_sc = np.array(ab)/scaling_factors
    # print(ab_sc)
    modelParameters = {'a2':ab_sc[0], 'a3': ab_sc[1], 'b2': ab_sc[2]}
    timeParameters = {'t_start':0, 't_stop':tspan[-1], 'steps':tspan[-1]}
    
    _, result_SDE_it = runEulerMaruyama_initializedNoise_x1interpolated(timeParameters, num_cells, x0, modelParameters, noise_mat, x1_expression_interpolated)
    # Dstack (time, x, cell)
    result_SDE_it = np.dstack(result_SDE_it)
    
    wd = 0
    for time in tspan:
        real = data_x2[str(time)]
        simulated = np.transpose(result_SDE_it[:,time,:])
        wd_x2 = wasserstein_distance_1D(real[:,1], simulated, wd_param[0], wd_param[1], wd_param[2])
        wd += wd_x2
    
    print(wd)
    return wd

###############################################################################
def sde_x2_cf_Scaled_InitNoise_l2(ab, x0, tspan, data_x2, num_cells, wd_param, scaling_factors, noise_mat, x1_expression_interpolated):
    # print(ab)
    ab_sc = np.array(ab)/scaling_factors
    # print(ab_sc)
    modelParameters = {'a2':ab_sc[0], 'a3': ab_sc[1], 'b2': ab_sc[2]}
    timeParameters = {'t_start':0, 't_stop':tspan[-1], 'steps':tspan[-1]}
    
    _, result_SDE_it = runEulerMaruyama_initializedNoise_x1interpolated(timeParameters, num_cells, x0, modelParameters, noise_mat, x1_expression_interpolated)
    # Dstack (time, x, cell)
    result_SDE_it = np.dstack(result_SDE_it)
    
    wd = 0
    for time in tspan:
        real = data_x2[str(time)]
        simulated = np.transpose(result_SDE_it[:,time,:])
        wd_x2 = wasserstein_distance_1D(real[:,1], simulated, wd_param[0], wd_param[1], wd_param[2])
        wd += wd_x2**2
    
    wd = np.sqrt(wd)
    print(wd)
    return wd

###############################################################################
def sde_x2_cf_Scaled_InitNoise_l2_df(ab, x0, tspan, data_x2, num_cells, wd_param, scaling_factors, noise_mat, x1_expression_interpolated):
    # print(ab)
    ab_sc = np.array(ab)/scaling_factors
    # print(ab_sc)
    modelParameters = {'a2':ab_sc[0], 'a3': ab_sc[1], 'b2': ab_sc[2]}
    timeParameters = {'t_start':0, 't_stop':tspan[-1], 'steps':tspan[-1]}
    
    _, result_SDE_it = runEulerMaruyama_initializedNoise_x1interpolated(timeParameters, num_cells, x0, modelParameters, noise_mat, x1_expression_interpolated)
    # Dstack (time, x, cell)
    result_SDE_it = np.dstack(result_SDE_it)
    
    wd = 0
    for time in tspan:
        real = data_x2[str(time)]
        simulated = np.transpose(result_SDE_it[:,time,:])
        wd_x2 = wasserstein_distance_1D(real, simulated, wd_param[0], wd_param[1], wd_param[2])
        wd += wd_x2**2
    
    wd = np.sqrt(wd)
    print(wd)
    return wd



