# -*- coding: utf-8 -*-
"""
Created on Mon May  2 09:16:29 2022

@author: MGR

Estimation Parameters SDE
"""

import os 
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy.optimize import minimize
from time import process_time

plt.rc('font', size=12)          # controls default text sizes
plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.rc('legend', fontsize=14)    # legend fontsize
plt.rc('figure', titlesize=20)  # fontsize of the figure title

from wasserstein_distance import wasserstein_distance_1D, wasserstein_distance_plots
from modelClass_ODE_SDE import  runEulerMaruyama_noplot, runEulerMaruyama_initializedNoise, dW_matrix

from sde_cost_functions import sde_cf_notScaled_notInitNoise_l1, sde_cf_notScaled_notInitNoise_l2, sde_cf_Scaled_notInitNoise_l1, sde_cf_Scaled_notInitNoise_l2
from sde_cost_functions import sde_cf_notScaled_InitNoise_l1, sde_cf_notScaled_InitNoise_l2, sde_cf_Scaled_InitNoise_l1, sde_cf_Scaled_InitNoise_l2

from sde_cost_functions_plots import plot_sde_data, plot_cf_1P


working_dir = 'C:/Users/Carlos/Desktop/Thesis'
matrices_directory = working_dir+'/Results/1_SyntheticData_Gillespie'
results_directory = working_dir+'/Results/2_SDE_ParameterEstimation'
os.chdir(working_dir)


#############################################################################
## DATA - Preload matrices of scRNA seq
#############################################################################

# # For snapshots:
times = ['1', '5', '10', '20', '40']

n_timepoints = len(times)

# For now I know it is always 1000 cells
expression = {}
colors = cm.get_cmap('tab10', 8).colors

# Plot them to see how they behave... 
fig = plt.figure(figsize=(20, 5))
ax1 = plt.subplot(1,2,1)
ax2 = plt.subplot(1,2,2)

for time in times:
    i = times.index(time)
    filename = matrices_directory+'/t'+time+'.xlsx'
    matrix = pd.read_excel(filename).to_numpy()
    expression[time]=matrix
    ax1.hist(matrix[:,0],bins=np.arange(0,40,1), histtype='stepfilled', color=colors[i], alpha=0.8, align='left')
    ax2.hist(matrix[:,1],bins=np.arange(0,40,1), histtype='stepfilled', color=colors[i], alpha=0.8, align='left')
    
    
ax1.set(title='X1', xlabel='mRNA molecules', ylabel='N° of cells', ylim=[0,400])
ax2.set(title='X2', xlabel='mRNA molecules', ylabel='N° of cells', ylim=[0,400])
ax1.legend(times, title='Time', loc="upper right")
ax2.legend(times, title='Time', loc="upper right")

plt.savefig(results_directory+'/Data_simulated_times_'+'_'.join(times)+'.png')
plt.show()
fig.clear()  


#############################################################################
# For plotting 

tspan = list(map(int, times))
x0 = {'x1': 10, 'x2': 10}
ab_exp= [1, 0.05, 0.1, 0.01, 0.1]

ab_names = ['a1', 'b1', 'a2', 'a3', 'b2']
ab_opt = [1, 0.05, 0.1, 0.01, 0.1]

scaling_factors = [1, 10, 1, 10, 10]
ab_opt_scaled = np.array(ab_opt)*scaling_factors



n_cells = 1000
timeParameters = {'t_start':0, 't_stop':tspan[-1], 'steps':tspan[-1]}

noise_matrix = dW_matrix(5, timeParameters, n_cells)
wd_params = [1, 50, 0.1]


#############################################################################


tspan = list(map(int, times))
x0 = {'x1': 10, 'x2': 10}
ab_exp= [1, 0.05, 0.1, 0.01, 0.1]


 
wd_parameters = [1, 50, 0.1] #sigma, Ntrunc, step

plot_sde_data(ab_exp, x0, tspan, expression, 1000, wd_parameters, results_directory)


ab1 = [1, 1, 1, 1, 1]
plot_sde_data(ab1, x0, tspan, expression, 1000, wd_parameters, results_directory)
ab2 = [0, 0, 0, 0, 0]
plot_sde_data(ab2, x0, tspan, expression, 1000, wd_parameters, results_directory)


#############################################################################
## Objective Function(s) - defined in sde_cost_functions.py



cost_functions = [sde_cf_notScaled_notInitNoise_l1, sde_cf_notScaled_notInitNoise_l2, sde_cf_Scaled_notInitNoise_l1, sde_cf_Scaled_notInitNoise_l2,
                  sde_cf_notScaled_InitNoise_l1, sde_cf_notScaled_InitNoise_l2, sde_cf_Scaled_InitNoise_l1, sde_cf_Scaled_InitNoise_l2]

cost_functions_scaled = [0, 0, 1, 1, 0, 0, 1, 1]

cost_functions_noise = [0, 0, 0, 0, 1, 1, 1, 1]

cost_functions_Lnorm = [1, 2, 1, 2, 1, 2, 1, 2]

#############################################################################
# Plot OF changing only 1 parameter at the time



tests = [20, 2, 0.2]
# tests = [2]
#tests = [0.2]
#tests = [20]

# Ready - 1000
for test in tests:
    par_try = np.linspace(0, test, 1000) 

    #plot_cf_1P(sde_cf_notScaled_notInitNoise_l1, 'notScaled_notInitNoise_l1', ab_opt, ab_names, par_try, x0, tspan, expression,  n_cells, wd_params, scaling_factors, noise_matrix, results_directory) 
    plot_cf_1P(sde_cf_notScaled_notInitNoise_l2, 'notScaled_notInitNoise_l2', ab_opt, ab_names, par_try, x0, tspan, expression,  n_cells, wd_params, scaling_factors, noise_matrix, results_directory) 


# Ready - missing all
for test in tests:
    par_try = np.linspace(0, test, 1000) 

    #plot_cf_1P(sde_cf_Scaled_notInitNoise_l1, 'Scaled_notInitNoise_l1', ab_opt_scaled, ab_names, par_try, x0, tspan, expression,  n_cells, wd_params, scaling_factors, noise_matrix, results_directory) 
    plot_cf_1P(sde_cf_Scaled_notInitNoise_l2, 'Scaled_notInitNoise_l2', ab_opt_scaled, ab_names, par_try, x0, tspan, expression,  n_cells, wd_params, scaling_factors, noise_matrix, results_directory) 

# Ready - 100
# Ready - 1000 test = 0.2
# Running 1000 2 in C1
for test in tests:
    par_try = np.linspace(0, test, 1000) 

    #plot_cf_1P(sde_cf_notScaled_InitNoise_l1, 'notScaled_InitNoise_l1', ab_opt, ab_names, par_try, x0, tspan, expression,  n_cells, wd_params, scaling_factors, noise_matrix, results_directory) 
    plot_cf_1P(sde_cf_notScaled_InitNoise_l2, 'notScaled_InitNoise_l2', ab_opt, ab_names, par_try, x0, tspan, expression,  n_cells, wd_params, scaling_factors, noise_matrix, results_directory) 

# Ready - Missing 2 and 02
for test in tests:
    par_try = np.linspace(0, test, 1000) 

    #plot_cf_1P(sde_cf_Scaled_InitNoise_l1, 'Scaled_InitNoise_l1', ab_opt_scaled, ab_names, par_try, x0, tspan, expression,  n_cells, wd_params, scaling_factors, noise_matrix, results_directory) 
    plot_cf_1P(sde_cf_Scaled_InitNoise_l1, 'Scaled_InitNoise_l2', ab_opt_scaled, ab_names, par_try, x0, tspan, expression,  n_cells, wd_params, scaling_factors, noise_matrix, results_directory) 


#par_try = np.linspace(0, 3, 10) 

#plot_cf_1P(sde_cf_Scaled_notInitNoise_l1, 'Scaled_notInitNoise_l1', ab_opt_scaled, ab_names, par_try, x0, tspan, expression,  n_cells, wd_params, scaling_factors, noise_matrix, results_directory) 
#plot_cf_1P(sde_cf_Scaled_notInitNoise_l2, 'Scaled_notInitNoise_l2', ab_opt_scaled, ab_names, par_try, x0, tspan, expression,  n_cells, wd_params, scaling_factors, noise_matrix, results_directory) 

#############################################################################
#############################################################################
## Parameter estimation


## Store the results
results = pd.DataFrame(columns = ['Solver', 'Param_ICs', 'Param_bounds', 'time', 'Scaled', 'Noise', 'Lnorm', 'OF', 'a1', 'b1', 'a2', 'a3', 'b2'])



# Initial Parameters - 1
ab1 = [0, 0, 0, 0, 0]

ab2 = [0.1, 0.1, 0.1, 0.1, 0.1]

ab3 = [1, 1, 1, 1, 1]

ab4 = [10, 10, 10, 10, 10]


# Bounds - pos
bnds0 = ((0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf))

# Bounds - 1
bnds1 = ((0, 1), (0, 1), (0, 1), (0, 1), (0, 1))


# Bounds - 2
bnds2 = ((0, 10), (0, 10), (0, 10), (0, 10), (0, 10))


# Bounds - 3
bnds3 = ((0, 100), (0, 100), (0, 100), (0, 100), (0, 100))



param_ICs = [ab1, ab2, ab3, ab4]
param_bnds = [bnds0, bnds1, bnds2, bnds3]




for param_IC in param_ICs:
    writer = pd.ExcelWriter(results_directory+'/SDE_param_est_methods_comparison'+str(param_IC[0])+'.xlsx', engine="openpyxl")
    for param_bound in param_bnds:
    
            for of in range(len(cost_functions)):
                #result = {'Solver': 'Min - L-NFGS-B ', 'Param_ICs': 'param_IC', 'Param_bounds': 'param_bound', 'time': 'time_elapsed',  'Scaled': 'cost_functions_scaled[of]', 'Noise': 'cost_functions_noise[of]', 'Lnorm': 'cost_functions_Lnorm[of]', 'OF':' OF_res', 'a1': 'a1_o', 'b1': 'b1_o', 'a2': 'a2_o', 'a3': 'a3_o', 'b2': 'b2_o'} 
                #results = results.append(result, ignore_index = True)

                
                try:               
                    start = process_time() 
                    res_min = minimize(cost_functions[of], param_IC, bounds=param_bound, args=(x0, tspan, expression, n_cells, wd_params, scaling_factors, noise_matrix), method = 'L-BFGS-B')
                    end = process_time()
                    time_elapsed = end - start
                    
                    a1_o, b1_o, a2_o, a3_o, b2_o = res_min.x
                    OF_res = res_min.fun
                    
                    
                    result = {'Solver': 'Min - L-NFGS-B ', 'Param_ICs': param_IC, 'Param_bounds': param_bound, 'time': time_elapsed,  'Scaled': cost_functions_scaled[of], 'Noise': cost_functions_noise[of], 'Lnorm': cost_functions_Lnorm[of], 'OF': OF_res, 'a1': a1_o, 'b1': b1_o, 'a2': a2_o, 'a3': a3_o, 'b2': b2_o} 
                    results = results.append(result, ignore_index = True)

                except:
                    result = {'Solver': 'Min - L-NFGS-B ', 'Param_ICs': param_IC, 'Param_bounds': param_bound, 'time': 0,  'Scaled': cost_functions_scaled[of], 'Noise': cost_functions_noise[of], 'Lnorm': cost_functions_Lnorm[of], 'OF': 0, 'a1': 0, 'b1': 0, 'a2': 0, 'a3': 0, 'b2': 0} 

    results.to_excel(writer, sheet_name = 'Results')  
    writer.save()  
    
    
    





















