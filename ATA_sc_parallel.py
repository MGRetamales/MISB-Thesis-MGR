# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 08:09:54 2022

@author: MGR 


ATA sc function
Writen with only one set of parameters (no fors inside)
so it can be parallelized
"""
from scipy.optimize import minimize
import numpy as np
from time import process_time


from gene_interpolation import interpolate_x1_df
from ATAsc_cost_functions import sde_x2_cf_Scaled_InitNoise_x1int_x2ICs_l2, sde_x2_cf_Scaled_InitNoise_alone_x2ICs_l2



def ATA_sc_x2ICs_pc(expression, ab_init, bound, wd_params, gene1, gene2, noise_matrix, solver):
    # Expression is a dictionary, that has index of numbers and content is DF in the shape genes x cells
    # AB init is ONE set of initial conditions
    # Bounds is ONE set of bounds for the parameters 
    # gene1, gene2 is ONE pair of genes to be tested
    # Get timepoints
    times = list(expression.keys())
    tspan = list(map(int, times))
    
    dt = 1
    
    # Initial time and initial conditions
    initial_timepoint = times[0]
    initial_conditions = expression[initial_timepoint]
    
    # Get list of all genes in the system 
    genes = list(initial_conditions.index)
    
    # See if elements of gene1 and gene2 exist
    # End unction if not
    if gene1 not in genes:
        #print(gene1 + ' is not on the list of genes for the system evaluated')
        result = {'Solver': solver, 'Param_ICs': ab_init, 'Param_bounds': bound, 'time': 0,  'Scaled': 1, 'Noise': 1, 'Lnorm': 2, 'gene1': gene1, 'gene2': gene2, 'OF': 0, 'a2': 0, 'a3': 0, 'b2': 0} 
        return result
    
    # End unction if not
    if gene2 not in genes:
        #print(gene2 + ' is not on the list of genes for the system evaluated')
        result = {'Solver': solver, 'Param_ICs': ab_init, 'Param_bounds': bound, 'time': 0,  'Scaled': 1, 'Noise': 1, 'Lnorm': 2, 'gene1': gene1, 'gene2': gene2, 'OF': 0, 'a2': 0, 'a3': 0, 'b2': 0} 
        return result
    
    
    # Loop through gene1 - now only one gene1
        
    # Do the interpolation for gene1
    gene_interpolated = interpolate_x1_df(expression, gene1, 'cubic', dt)
    gene_x1_interpolated = gene_interpolated*(gene_interpolated >= 0)
    
    
    # For ICs of the rest of the genes
    # # Sort timepoint1 by gene 1

    data_x0_sorted = initial_conditions.sort_values(by = gene1, axis = 1)
    
    # Loop through gene2 - now only one gene2
    # if gene1 is the same as gene2, don't look for the regulation
    if gene1 == gene2:
       # j += 1
        result = {'Solver': solver, 'Param_ICs': ab_init, 'Param_bounds': bound, 'time': 0,  'Scaled': 1, 'Noise': 1, 'Lnorm': 2, 'gene1': gene1, 'gene2': gene2, 'OF': 0, 'a2': 0, 'a3': 0, 'b2': 0} 
    
    else:
        # print(gene1, gene2) 
        
        # Save data of gene2 for the timepoints
        data_gene2 = {}
        for time in times:
            data_gene2_time = expression[time].loc[gene2,:]
            data_gene2[time] = data_gene2_time
       
        # Get the Initial conditions for X1 and X2 for the same cell 
        x1_IC = data_x0_sorted.loc[gene1]
        x2_IC = data_x0_sorted.loc[gene2]
        
        # Get the mean of the initial condition, to be used for scaling purposes
        mean_IC_x1 = np.mean(x1_IC)
        mean_IC_x2 = np.mean(x2_IC)
        # scaling factor [a2, a3*x1, b2*x3]
        scaling_factors = [1, mean_IC_x1, mean_IC_x2]
        
        
        # print(gene1, gene2, bound, ab_IC)
        # Run optimization to estimate parameters a2, a3 and b2
        start = process_time()
        res_min = minimize(sde_x2_cf_Scaled_InitNoise_x1int_x2ICs_l2, ab_init, bounds=bound, args=(x2_IC, tspan, data_gene2, wd_params, 
                                                                                  scaling_factors, noise_matrix, gene_x1_interpolated), 
                          method = solver)
        end = process_time()
        time_elapsed = end - start
        
        # Save the results
        a2_o, a3_o, b2_o = res_min.x
        OF_res = res_min.fun
        #results_OF[i,j] = OF_res
        result = {'Solver': solver, 'Param_ICs': ab_init, 'Param_bounds': bound, 'time': time_elapsed,  'Scaled': 1, 'Noise': 1, 'Lnorm': 2, 'gene1': gene1, 'gene2': gene2, 'OF': OF_res, 'a2': a2_o, 'a3': a3_o, 'b2': b2_o} 


    return result



# Is a linear approximation of x2 without x1 better? 
# For evaluation purposes
def ATA_sc_x2_alone_pc(expression, ab_init, bound, wd_params, gene2, noise_matrix, solver):
    # Expression is a dictionary, that has index of numbers and content is DF in the shape genes x cells
    # AB init is ONE set of initial conditions
    # Bounds is ONE set of bounds for the parameters 
    # gene2 is ONE gene to be tested
    
    # Get timepoints
    times = list(expression.keys())
    tspan = list(map(int, times))
    
    # Initial time and initial conditions
    initial_timepoint = times[0]
    initial_conditions = expression[initial_timepoint]
    
    # Get list of all genes in the system 
    genes = list(initial_conditions.index)
    
    # See if gene2 exist ont he list of genes
    

    # End unction if not
    if gene2 not in genes:
        print(gene2 + ' is not on the list of genes for the system evaluated')
        result = {'Solver': solver, 'Param_ICs': ab_init, 'Param_bounds': bound, 'time': 0,  'Scaled': 1, 'Noise': 1, 'Lnorm': 2, 'gene1': 0, 'gene2': gene2, 'OF': 0, 'a2': 0, 'a3': 0, 'b2': 0} 
        return result
    

    # Loop through gene2
    
    # Save data of gene2 for the timepoints
    data_gene2 = {}
    for time in times:
        data_gene2_time = expression[time].loc[gene2,:]
        data_gene2[time] = data_gene2_time
   
    # Get the Initial conditions for X2
    x2_IC = initial_conditions.loc[gene2]
    # n_cells = len(x2_IC)
    
    # Get the mean of the initial condition, to be used for scaling purposes
    mean_IC_x2 = np.mean(x2_IC)
    # scaling factor [a2, a3*x1, b2*x3]
    scaling_factors = [1, mean_IC_x2]
    
    # Initialize the noise matrix - run the OF with the same noise each time
    # noise_matrix = dW_matrix(3, timeParameters, n_cells) 
    
    # Run optimization to estimate parameters a2, a3 and b2

    # print(gene2, bound, ab_init)
    start = process_time()
    res_min = minimize(sde_x2_cf_Scaled_InitNoise_alone_x2ICs_l2, ab_init, bounds=bound, args=(x2_IC, tspan, data_gene2, wd_params, 
                                                                              scaling_factors, noise_matrix), 
                      method = solver)
    end = process_time()
    time_elapsed = end - start
    
    # Save the results
    a2_o, b2_o = res_min.x
    OF_res = res_min.fun
    #results_OF[j] = OF_res
    result = {'Solver': solver, 'Param_ICs': ab_init, 'Param_bounds': bound, 'time': time_elapsed,  'Scaled': 1, 'Noise': 1, 'Lnorm': 2, 'gene1': 0, 'gene2': gene2, 'OF': OF_res, 'a2': a2_o, 'a3': 0, 'b2': b2_o} 
    
    # return results_OF, results
    return result




















