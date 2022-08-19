# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 08:30:23 2022

@author: MGR

ATA-sc - from the parallel computing functions
- read simulation from gillespie
- run opptimization algoritm to estimate parameters (like - 2_SDE_paramEstimation.py)
- Run it for all genes, alone without regulation

"""

import os 
import numpy as np
import pandas as pd
import time
import itertools
from multiprocessing import Pool

from modelClass_ODE_SDE_x2 import  dW_matrix
from ATA_sc_parallel import ATA_sc_x2_alone_pc


    

#############################################################################
## Run the sc-ATA algorithm with NO regulation

## Function of minimization
def f_min(fun_parameters):
    expression = fun_parameters[0]
    param_IC = fun_parameters[1]
    bound = fun_parameters[2]
    wd_params = fun_parameters[3]
    gene2 = fun_parameters[4]
    dw_matrix = fun_parameters[5]
    solver = fun_parameters[6]
    
    result = ATA_sc_x2_alone_pc(expression, param_IC, bound, wd_params, gene2, dw_matrix, solver)
    return result





if __name__ == '__main__':
    
    #############################################################################
    ## DIRECTORY - set directories and names for files
    #############################################################################
    
    
    ## Important directory, has to be changed in each computer
    working_dir = 'C:/Users/Carlos/Desktop/Thesis'
    os.chdir(working_dir)
    
    network_names = ['network02_04']
    #network_name = 'network05_03'
    
    n_processors = 2
    
    for network_name in network_names:
        genes2 = 'all'
        np.random.seed(10)

        
        #network_name = input("Enter name of network:")
        #network_name = 'network01'
        #print("Name of network: "+network_name)
    
    
        matrices_directory = working_dir+'/Results/00_Whole_Pipelines/'+network_name+'/0_Data'
    
        results_directory = working_dir+'/Results/00_Whole_Pipelines/'+network_name+'/3_ATAsc'
        if not os.path.isdir(results_directory):
            os.mkdir(results_directory)
            
            
            
        #############################################################################
        ## DATA - Preload matrices of scRNA seq
        #############################################################################
    
    
        # # Time of snapshots:
        files =  os.listdir(matrices_directory)
        files = list(filter(lambda x: '.txt' in x, os.listdir(matrices_directory)))
        times = [files[i][1:-4] for i in range(len(files))]
        times.sort(key= int)
    
        # Genes in the first file
        file_0_name = matrices_directory+'/t'+times[0]+'.txt'
        matrix = pd.read_csv(file_0_name, sep = ' ', index_col=0)
        genes = list(matrix.index)
    
        print('The dataset contains the following genes:')
        print(genes)
        print('We will evaluate how gene2 can regulate itself without any gene2 regulation')
        print('Enter target list (gene2)')
    
        # Genes to analyze
        # genes2 = input("Enter target list separated by comma. If you want all the genes, write ALL: ")
        #genes2 = 'all'
    
    
        if genes2.upper() == 'ALL':
            genes2 = genes[:]
        else: 
            genes2 = genes2.replace(" ", "")
            genes2 = genes2.split(',')
    
    
        print('All the data will be saved in: '+results_directory)
        print('Loading data from: '+ matrices_directory)
    
        # Save the expression o each time point
        # For now it has to be 1000 cells in each timepoint
        expression = {}
        max_expression = 10
        for time_it in times:
            i = times.index(time_it)
            filename = matrices_directory+'/t'+time_it+'.txt'
            matrix = pd.read_csv(filename, sep = ' ', index_col=0)
            
            max_expression = max(matrix.max().max()+10, max_expression) 
            expression[time_it]=matrix
        
        # Initilize noise matrix - ONCE
        tspan = list(map(int, times))
        
        timeParameters = {'t_start':tspan[0], 't_stop':tspan[-1], 'steps':tspan[-1]}
        dw_matrix = dW_matrix(3, timeParameters, 1000) 
    
        wd_params = [1, max_expression, 0.1]
        
        #############################################################################
        ## ATAsc - Run ATA for each gene regulating itself only
        #############################################################################
        
        
        # Initial Parameters - 1
        ab1 = [0, 0]
        ab2 = [0.1, 0.1]
        ab3 = [1, 1]
        
        # Bounds - 1
        bnds1 = ((0, 1), (0, 1))
        
        param_ICs = [ab1, ab2, ab3]
        param_bnds = [bnds1]
        
        solver = 'L-BFGS-B'
        expression_l = [expression]
        wd_params_l = [wd_params]
        solver_l = [solver]
        dw_matrix_l = [dw_matrix]
        
        param_conds_g = list(itertools.product(expression_l, param_ICs, param_bnds, wd_params_l, genes2, dw_matrix_l, solver_l))
        
        
        TimeSTARTrunOptim = time.time()
        with Pool(n_processors) as p:
            res = list(p.map(f_min, param_conds_g))
        TimeSTOPrunOptim = time.time()
        TimeOneRunOptim = (TimeSTOPrunOptim-TimeSTARTrunOptim)
        print('\n Time for running once all the PROCESSES in parallel with '+str(n_processors)+' workers is (s): ' +  str(TimeOneRunOptim))
    
        
        results_df_alone = pd.DataFrame(res)
    
    
        genes2_str = '_'.join(genes2)
        timestp = str(int(time.time()))
        filename = '/NOREG_resultsATAsc_t_'+timestp+'.xlsx'
        results_df_alone.to_excel(results_directory+filename)


