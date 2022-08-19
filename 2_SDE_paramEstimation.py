# -*- coding: utf-8 -*-
"""
Created on Mon May  17 21:16:29 2022

@author: MGR

Estimation Parameters SDE
Only works for 2 genes at the time

"""

import os 
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy.optimize import minimize
from time import process_time


working_dir = 'C:/Users/Carlos/Desktop/Thesis'
os.chdir(working_dir)


from modelClass_ODE_SDE_x2 import  dW_matrix
from gene_interpolation import interpolate_x1_df

# Objective funtions - with and without regultion from x1 (gene1)
from Thesis_project.utilities_pipeline.ATAsc_cost_functions import sde_x2_cf_Scaled_InitNoise_x1int_x2ICs_l1, sde_x2_cf_Scaled_InitNoise_x1int_x2ICs_l2
from Thesis_project.utilities_pipeline.ATAsc_cost_functions import sde_x2_cf_Scaled_InitNoise_alone_x2ICs_l1, sde_x2_cf_Scaled_InitNoise_alone_x2ICs_l2



network_name = input("Enter name of network:")
print("Name of network: "+network_name)


matrices_directory = working_dir+'/Results/00_Whole_Pipelines/'+network_name+'/0_Data'


results_directory = working_dir+'/Results/00_Whole_Pipelines/'+network_name+'/2_SDE_paramEstimation'
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
print('We will evaluate how gene1 regulates gene2')
print('Enter gene1 and gene2!')

# Genes to analyze
gene1 = input("Enter gene1:")
gene2 = input("Enter gene2:")

    
    
print('We will evaluate how '+gene1+ ' regulates '+gene2)

results_directory = results_directory+'/'+gene1+'_'+gene2
if not os.path.isdir(results_directory):
    os.mkdir(results_directory)
    
print('All the data will be saved in: '+results_directory)

print('Loading data from: '+ matrices_directory)
# For now I know it is always 1000 cells
expression = {}
colors = cm.get_cmap('Paired', len(times)).colors

# Plot them to see how they behave... 
fig = plt.figure(figsize=(20, 5))
ax1 = plt.subplot(1,2,1) # gene1
ax2 = plt.subplot(1,2,2) # gene2

max_expression = 10
for time in times:
    i = times.index(time)
    filename = matrices_directory+'/t'+time+'.txt'
    matrix = pd.read_csv(filename, sep = ' ', index_col=0)
    
    # filename = matrices_directory+'/t'+time+'.xlsx'
    # matrix = pd.read_excel(filename).to_numpy()
    matrix = matrix.loc[[gene1, gene2]]
    max_expression = max(matrix.max().max()+10, max_expression) 
    expression[time]=matrix
    
    ax1.hist(matrix.loc[gene1,:],bins=np.arange(0,100,1), histtype='stepfilled', color=colors[i], alpha=0.5, align='left')
    ax2.hist(matrix.loc[gene2,:],bins=np.arange(0,100,1), histtype='stepfilled', color=colors[i], alpha=0.5, align='left')
    
    
ax1.set(title=gene1, xlabel='mRNA molecules', ylabel='N° of cells')
ax2.set(title=gene2, xlabel='mRNA molecules', ylabel='N° of cells')
ax1.legend(times, title='Time', loc="upper right")
ax2.legend(times, title='Time', loc="upper right")

plt.savefig(results_directory+'/Data_simulated_times_'+'_'.join(times)+'.png')
plt.show()
fig.clear()  

print('Data plot saved as: '+results_directory +'/Data_simulated_times_'+'_'.join(times)+'.png')


tspan = list(map(int, times))
timeParameters = {'t_start':0, 't_stop':tspan[-1], 'steps':tspan[-1]}
wd_params = [1, max_expression, 0.1]

#############################################################################
## Objective Function(s) - defined in sde_cost_functions.py
print('Loading optimization functions to try')
# With x1 regulation
#cost_functions = [sde_x2_cf_Scaled_InitNoise_x1int_x2ICs_l1, sde_x2_cf_Scaled_InitNoise_x1int_x2ICs_l2]
cost_functions = [sde_x2_cf_Scaled_InitNoise_x1int_x2ICs_l2]

# Without x1 regulation
#cost_functions_alone = [sde_x2_cf_Scaled_InitNoise_alone_x2ICs_l1, sde_x2_cf_Scaled_InitNoise_alone_x2ICs_l2]
cost_functions_alone = [sde_x2_cf_Scaled_InitNoise_alone_x2ICs_l2]

# They are scaled, initialized noise and we will try L1 and L2 norms to compare
cost_functions_scaled = [1, 1]
cost_functions_noise = [1, 1]
cost_functions_Lnorm = [1, 2]


# Interpolation of gene1, to have its expression over time, not just snapshots
inter_type = 'cubic'
print('Interpolating gene1 with '+inter_type+' interpolation')
dt = 1
x_interpolated = interpolate_x1_df(expression, gene1, inter_type, dt)
x_interpolated = x_interpolated*(x_interpolated >= 0)


# Get Initial conditions
print('Setting initial conditions and sorting by the vlaue of x1')
initial_timepoint = times[0]
initial_conditions = expression[initial_timepoint]
# For ICs of the rest of the genes
# # Sort timepoint1 by gene 1
data_x0_sorted = initial_conditions.sort_values(by = gene1, axis = 1)


# Save data of gene2 for the timepoints
data_gene2 = {}
for time in times:
    data_gene2_time = expression[time].loc[gene2,:]
    data_gene2[time] = data_gene2_time

# Get the Initial conditions for X1 and X2 for the same cell 
x1_IC = data_x0_sorted.loc[gene1]
x2_IC = data_x0_sorted.loc[gene2]
n_cells = len(x2_IC)

# Get the mean of the initial condition, to be used for scaling purposes
mean_IC_x1 = np.mean(x1_IC)
mean_IC_x2 = np.mean(x2_IC)
# scaling factor [a2, a3*x1, b2*x3]
scaling_factors = [1, mean_IC_x1, mean_IC_x2]
# Initialize the noise matrix - run the OF with the same noise each time
noise_matrix = dW_matrix(3, timeParameters, n_cells) 


#############################################################################
#############################################################################
## Parameter estimation of X1 regulating X2


## Store the results
results = pd.DataFrame(columns = ['Solver', 'Param_ICs', 'Param_bounds', 'time', 'Scaled', 'Noise', 'Lnorm', 'OF', 'Gene1', 'Gene2', 'a2', 'a3', 'b2'])





# Initial Parameters - 1
ab1 = [0, 0, 0]

ab2 = [0.1, 0.1, 0.1]

ab3 = [1, 1, 1]

ab4 = [ 10, 10, 10]


# Bounds - pos
bnds0 = ((0, np.inf), (0, np.inf), (0, np.inf))

# Bounds - 1
bnds1 = ((0, 1), (0, 1), (0, 1))

# Bounds - 2
bnds2 = ((0, 10), (0, 10), (0, 10))


# Bounds - 3
bnds3 = ((0, 100), (0, 100), (0, 100))



param_ICs = [ab1, ab2, ab3, ab4]
param_bnds = [bnds0, bnds1, bnds2, bnds3]

#param_ICs = [ab1]
#param_bnds = [bnds0]



for param_IC in param_ICs:
    writer = pd.ExcelWriter(results_directory+'/SDE_x2_param_est_methods_comparison'+str(param_IC[0])+'.xlsx', engine="openpyxl")
    for param_bound in param_bnds:
    
            for of in range(len(cost_functions)):
                
                try:               
                    print(param_IC, param_bound, of)
                    start = process_time() 
                    # Run optimization to estimate parameters a2, a3 and b2
                    res_min = minimize(cost_functions[of], param_IC, bounds=param_bound, args=(x2_IC, tspan, data_gene2, wd_params, 
                                                                                              scaling_factors, noise_matrix, x_interpolated), 
                                      method = 'L-BFGS-B')
                    end = process_time()
                    time_elapsed = end - start
                    
                    a2_o, a3_o, b2_o = res_min.x
                    OF_res = res_min.fun
                    
                    
                    result = {'Solver': 'Min - L-NFGS-B ', 'Param_ICs': param_IC, 'Param_bounds': param_bound, 'time': time_elapsed,  'Scaled': cost_functions_scaled[of], 'Noise': cost_functions_noise[of], 'Lnorm': cost_functions_Lnorm[of], 'OF': OF_res, 'Gene1': gene1, 'Gene2': gene2, 'a2': a2_o, 'a3': a3_o, 'b2': b2_o} 
                    results = results.append(result, ignore_index = True)

                except:
                    result = {'Solver': 'Min - L-NFGS-B ', 'Param_ICs': param_IC, 'Param_bounds': param_bound, 'time': 0,  'Scaled': cost_functions_scaled[of], 'Noise': cost_functions_noise[of], 'Lnorm': cost_functions_Lnorm[of], 'OF': 0, 'a1': 0, 'b1': 0, 'a2': 0, 'a3': 0, 'b2': 0} 

    results.to_excel(writer, sheet_name = 'Results')  
    writer.save()  

writer = pd.ExcelWriter(results_directory+'/SDE_x2_param_est_methods_comparison_'+gene1+'_'+gene2+'.xlsx', engine="openpyxl")
results.to_excel(writer, sheet_name = 'Results') 
writer.save() 
writer.close()





#############################################################################
#############################################################################
## Parameter estimation of X2 without X1
scaling_factors = [1, mean_IC_x2]

## Store the results
results_alone = pd.DataFrame(columns = ['Solver', 'Param_ICs', 'Param_bounds', 'time', 'Scaled', 'Noise', 'Lnorm', 'OF', 'Gene1', 'Gene2', 'a2', 'a3', 'b2'])



# Initial Parameters - 1
ab1 = [0, 0]

ab2 = [0.1, 0.1]

ab3 = [1, 1]

ab4 = [ 10, 10]


# Bounds - pos
bnds0 = ((0, np.inf), (0, np.inf))

# Bounds - 1
bnds1 = ((0, 1), (0, 1))

# Bounds - 2
bnds2 = ((0, 10), (0, 10))


# Bounds - 3
bnds3 = ((0, 100), (0, 100))



param_ICs = [ab1, ab2, ab3, ab4]
param_bnds = [bnds0, bnds1, bnds2, bnds3]


scaling_factors = [1, mean_IC_x2]

# Initialize the noise matrix - run the OF with the same noise each time

# Run optimization to estimate parameters a2, a3 and b2

#param_ICs = [ab1]
#param_bnds = [bnds0]

for param_IC in param_ICs:
    writer = pd.ExcelWriter(results_directory+'/SDE_x2_param_est_methods_comparison_alone'+str(param_IC[0])+'.xlsx', engine="openpyxl")
    for param_bound in param_bnds:
    
            for of in range(len(cost_functions_alone)):
                
                try:
                    print(param_IC, param_bound, of)
                    start = process_time() 
                    # Run optimization to estimate parameters a2, a3 and b2
                    res_min = minimize(cost_functions_alone[of], param_IC, bounds=param_bound, args=(x2_IC, tspan, data_gene2, wd_params, 
                                                                                              scaling_factors, noise_matrix), 
                                      method = 'L-BFGS-B')

                    end = process_time()
                    time_elapsed = end - start
                    
                    a2_o, b2_o = res_min.x
                    OF_res = res_min.fun
                    
                    
                    result = {'Solver': 'Min - L-NFGS-B ', 'Param_ICs': param_IC, 'Param_bounds': param_bound, 'time': time_elapsed,  'Scaled': cost_functions_scaled[of], 'Noise': cost_functions_noise[of], 'Lnorm': cost_functions_Lnorm[of], 'OF': OF_res, 'Gene1': 0, 'Gene2': gene2, 'a2': a2_o, 'a3': 0, 'b2': b2_o} 
                    results_alone = results_alone.append(result, ignore_index = True)

                except:
                    result = {'Solver': 'Min - L-NFGS-B ', 'Param_ICs': param_IC, 'Param_bounds': param_bound, 'time': 0,  'Scaled': cost_functions_scaled[of], 'Noise': cost_functions_noise[of], 'Lnorm': cost_functions_Lnorm[of], 'OF': 0, 'Gene1': 0, 'Gene2': gene2, 'a2': 0, 'a3': 0, 'b2': 0} 
                    results_alone = results_alone.append(result, ignore_index = True)
                    
    results_alone.to_excel(writer, sheet_name = 'Results')  
    writer.save()  

writer = pd.ExcelWriter(results_directory+'/SDE_x2_param_est_methods_comparison_alone_'+gene2+'.xlsx', engine="openpyxl")
results_alone.to_excel(writer, sheet_name = 'Results') 
writer.save() 
writer.close()