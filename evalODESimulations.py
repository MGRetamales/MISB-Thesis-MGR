# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 14:29:38 2022

@author: MGR

Method Evaluation - Errors and time steps
"""
import os 
import numpy as np
import matplotlib.pyplot as plt
from time import process_time
import pandas as pd

from modelClass_ODE_SDE import runODE_model, runEulerMaruyama, runEulerMaruyama_noplot
from analyticalSolSystem import analyticalSolSystem


working_dir = 'C:/Users/Carlos/Desktop/Thesis'
#working_dir = 'C:/Users/Carlos/Desktop/Thesis/0_pipeline'
images_directory = working_dir+'/Results/0_Methods_SyntheticData/EvalSimulations'
os.chdir(working_dir)

plt.rc('font', size=12)          # controls default text sizes
plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.rc('legend', fontsize=14)    # legend fontsize
plt.rc('figure', titlesize=20)  # fontsize of the figure title


modelParameters = {'a1': 1, 'b1': 0.05, 'a2': 0.1, 'a3': 0.01, 'b2': 0.1}
ICs = {'x1': 10, 'x2': 10}
timeParameters = {'t_start':0, 't_stop':1000, 'steps':10001}

n_time_snaps = 3
N_cells_killed_per_snapshot = 1000
num_cells = n_time_snaps*N_cells_killed_per_snapshot


###############################################################################
# 0. Run analytical solution to the system: 
    
time_eval, x1_sol, x2_sol = analyticalSolSystem(timeParameters, ICs, modelParameters, True, images_directory)

# Fixed points 
fixed_x1 = modelParameters['a1']/modelParameters['b1']
fixed_x2 = modelParameters['a2']/modelParameters['b2'] + modelParameters['a3']/modelParameters['b2'] * modelParameters['a1']/modelParameters['b1']

# According to the plot, the fixed point is achieved around time 200
# Then we change the time range to focus on the transient state
timeParameters = {'t_start':0, 't_stop':200, 'steps':1000001}
time_eval, x1_sol, x2_sol = analyticalSolSystem(timeParameters, ICs, modelParameters, False, images_directory)
x_sol = np.transpose(np.array([x1_sol, x2_sol]))

###############################################################################
# 1. Run ODE - for the same time parameters as the analytical solution
solvers = ['RK45', 'RK23', 'DOP853',  'Radau', 'BDF', 'LSODA']

results_ODE = pd.DataFrame(columns=['Solver', 'Error_X1', 'Error_X2']) 

for solver in solvers:
    n_ODE_sim = 1
    time_ODE, result_ODE = runODE_model(timeParameters, n_ODE_sim, ICs, modelParameters, False, images_directory, solver)
    time_ODE = time_ODE[0]
    result_ODE = np.transpose(result_ODE[0])
    x1_ode = result_ODE[:,0]
    x2_ode = result_ODE[:,1]
    
    
    # Plot Analytical solution and ODE numerical integration solution
    
    fig = plt.figure()
    plt.plot(time_eval, x1_sol, '-', label = 'X1 - Analytical solution')
    plt.plot(time_eval, x2_sol, '-', label = 'X2 - Analytical solution')
    plt.plot(time_ODE, x1_ode, '--', label = 'X1 - ODE numerical integration')
    plt.plot(time_ODE, x2_ode, '--', label = 'X2 - ODE numerical integration')
    
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('N of Molecules')
    plt.title('Numerical vs Analytical solution of the system')
    plt.savefig(images_directory+'/ANA_vs_NumODE')
    
    plt.show()
    fig.clear()
    
    # Diferences with analytival solution:
    
    compare = np.transpose(np.array([abs(x1_sol - x1_ode)/x1_sol * 100, abs(x2_sol - x2_ode)/x2_sol * 100]))
    compare_x1 = abs(x1_sol - x1_ode)/x1_sol * 100
    compare_x2 = abs(x1_sol - x1_ode)/x1_sol * 100
    
    
    # mean error
    error_mean = np.mean(compare, axis = 0)
    error_mean_x1 = error_mean[0]
    error_mean_x2 = error_mean[1]
    error_mean = np.mean(error_mean)
    
    result_ODE = {'Solver': solver, 'Error_X1':error_mean_x1 , 'Error_X2': error_mean_x2 } 
    results_ODE = results_ODE.append(result_ODE, ignore_index = True)
    
    fig = plt.figure()
    plt.plot(time_eval, compare[:,0], label = 'X1')
    plt.plot(time_eval, compare[:,1], label = 'X2')
    plt.legend()
    plt.xlabel('Time')
    plt.xlim([0,200])
    plt.ylim([0,0.8])
    plt.ylabel('% Error')
    plt.title('Analytical solution vs Numerical \n approximation with '+ solver)
    plt.savefig(images_directory+'/ANA_vs_NumODE_Error_'+solver,  bbox_inches='tight' )
    
    plt.show()
    fig.clear()



###############################################################################

# 2. Run SDE - for diff time steps & diff numbers of trajectories
# Compare them with the analytical solution

colors = ['r', 'b', 'g', 'c', 'm', 'y']



delta_t_try = [10**-2, 10**-1, 10**-0, 10**+1]
number_trajectories = [1, 5, 10, 50, 100, 500, 1000, 5000]
n_runs = 3

results_sims = pd.DataFrame(columns=['Run', 'Delta_t', 'N_Sim', 'Time_Run', 'Error_Mean', 'Error_X1', 'Error_X2']) 


for run in range(n_runs):
    i = 0
    for num in number_trajectories:
        fig = plt.figure(figsize=(10, 8), constrained_layout=True)
        #fig.tight_layout()
        
        
        fig.suptitle('Mean of '+str(num)+' trajectories', fontsize=20)
         
        ax1 = plt.subplot(211)
        #ax1.plot(time_eval, x1_sol)
        ax1.set_title('X1')    
        ax1.set_xlim([0,200])
        ax1.set_ylabel('N of molecules')
        ax1.set_xlabel('Time')
            
        
        ax2 = plt.subplot(212)
        #ax2.plot(time_eval, x2_sol)
        ax2.set_title('X2')
        ax2.set_xlim([0,200])
        ax2.set_ylabel('N of molecules')
        ax2.set_xlabel('Time')
        
        # ax3 = plt.subplot(313)
        # ax3.set_title('% Error')
        # ax3.set_xlim([0,200])
        # ax3.set_ylim([0,200])
        # ax3.set_xlabel('Time')
        # ax3.set_ylabel('% Error')
        
        
        j = 0
        for delta_t in delta_t_try:
            steps = (200-0)/delta_t
            timeParameters = {'t_start':0, 't_stop':200, 'steps':int(steps)}
            
            start = process_time() 
            time_SDE, result_SDE = runEulerMaruyama_noplot(timeParameters, num, ICs, modelParameters)
            
            end = process_time()
            time_elapsed = end - start
            print(time_elapsed)
            
            x1 = np.zeros([len(time_SDE)])
            x2 = np.zeros([len(time_SDE)])
        
            for result in result_SDE:
                x1 += result[:,0]
                x2 += result[:,1]
            x1 = x1/num
            x2 = x2/num
            
            result_SDE_mean = np.array([x1,x2]).transpose()
            
            # Intersection of Elements of time:
            times_intersection = list(set(time_eval).intersection(set(time_SDE)))
            times_intersection.sort()
            # ID of elements interseced in ODE
            time_int_index = [list(np.where(time_eval  == x)[0]) for x in times_intersection]
            # ID of elements interseced in SDE
            time_int_SDE_index = [list(np.where(time_SDE  == x)[0]) for x in times_intersection]    
    
            # Compare
            selected = x_sol[time_int_index,:]
            selected = np.stack(selected, axis = 1)
            
            selected_SDE = result_SDE_mean[time_int_SDE_index,:]
            selected_SDE = np.stack(selected_SDE, axis = 1)
            
            # comparison
            compare = abs(selected-selected_SDE)/selected * 100
            
            
            # mean error
            error_mean = np.mean(compare, axis = 1)
            error_mean_x1 = error_mean[0,0]
            error_mean_x2 = error_mean[0,1]
            error_mean = np.mean(error_mean)
            
            
            # accum error
            error_acc = np.cumsum(compare, axis = 1)
            # ax3.plot(times_intersection, compare[0,:,0], colors[j], linestyle = 'dashed')
            # ax3.plot(times_intersection, compare[0,:,1], colors[j])
            
            
            ax1.plot(time_SDE, x1, colors[j])
            ax2.plot(time_SDE, x2, colors[j])
            
            result_sim = {'Run': run, 'Delta_t': delta_t, 'N_Sim': num, 'Time_Run': time_elapsed , 'Error_Mean':error_mean , 'Error_X1':error_mean_x1 , 'Error_X2': error_mean_x2 } 
            results_sims = results_sims.append(result_sim, ignore_index = True)
            j += 1
        
        # ax3.plot([0, 0], [0, 0], 'k', linestyle = 'dashed', label = 'x1')
        # ax3.plot([0, 0], [0, 0], 'k', label = 'x2')
        # ax3.legend(bbox_to_anchor=(1.0, 1.0))
        
        ax1.plot(time_eval, x1_sol , 'm', linewidth=3.0)
        ax2.plot(time_eval, x2_sol, 'm', linewidth=3.0)
        
        leg = ['Analytical']
        deltas_leg = ['SDE dt: ' + str(x) for x in delta_t_try]
        deltas_leg.extend(leg)
        ax1.legend(deltas_leg, bbox_to_anchor=(1.0, 1.0))
        
        name_plot = ('/Trajectories_error_N_'+str(num)+'_run_'+str(run)+'.png')
        plt.savefig(images_directory+name_plot)
        
        plt.show()
        
        i += 1 

writer = pd.ExcelWriter(images_directory+'/SDE_N_Comparison_S1to1000_dt.xlsx', engine="openpyxl")
results_sims.to_excel(writer, sheet_name = 'Simulations')  


stat_dt_keys = results_sims['Delta_t'].unique()


results_sims_stat = results_sims.groupby(['Delta_t', 'N_Sim']).agg(['mean', 'std'])
results_sims_stat.to_excel(writer, sheet_name = 'Stats')  

writer.save()  



# Time Run
fig, ax = plt.subplots()

i = 0
for dt_key in stat_dt_keys:
    print(dt_key)
    res_filtered = results_sims_stat.loc[dt_key]
    res_filtered['Time_Run', 'mean'].plot(ax = ax, kind='line', marker='o', 
                                            yerr=res_filtered['Time_Run', 'std'].values.T,
                                            label=dt_key, color = colors[i])
    i += 1
 
plt.xlabel('Number of Cells')
plt.ylabel('Running time [s]')    
plt.title('Simulation time for different integration \n step sizes and number of cells')
plt.yscale('log') 
plt.xscale('log')    
plt.legend(loc='best', title="Numerical Integration Step size")

name_plot = ('/Time_run.png')      
plt.savefig(images_directory+name_plot, bbox_inches='tight')
#plt.show()


# Mean Error
fig, ax = plt.subplots()

i = 0
for dt_key in stat_dt_keys:
    print(dt_key)
    res_filtered = results_sims_stat.loc[dt_key]
    res_filtered['Error_Mean', 'mean'].plot(ax = ax, kind='line', marker='o', 
                                            yerr=res_filtered['Error_Mean', 'std'].values.T,
                                            label=dt_key, color = colors[i])
    i += 1


plt.xlabel('Number of Cells')
plt.ylabel('Error %') 
plt.title('Simulation Mean Error when \n compared with Analytical Solution')
plt.yscale('log') 
plt.xscale('log')    
plt.legend(loc='best', title="Numerical Integration Step size")

name_plot = ('/Error_mean.png')      
plt.savefig(images_directory+name_plot, bbox_inches='tight')
#plt.show()




# X1 Error
fig, ax = plt.subplots()
i = 0
for dt_key in stat_dt_keys:
    print(dt_key)
    res_filtered = results_sims_stat.loc[dt_key]
    res_filtered['Error_X1', 'mean'].plot(ax = ax, kind='line', marker='o', 
                                            yerr=res_filtered['Error_X1', 'std'].values.T,
                                            label=dt_key, color = colors[i])
    i += 1

plt.xlabel('Number of Cells')
plt.ylabel('Error X1 %') 
plt.title('Simulation Mean Error for X1 when \n compared with Analytical Solution')
plt.yscale('log') 
plt.xscale('log')    
plt.legend(loc='best', title="Numerical Integration Step size")

name_plot = ('/error_x1.png')      
plt.savefig(images_directory+name_plot, bbox_inches='tight')
#plt.show()




# X2 Error
fig, ax = plt.subplots()
i = 0
for dt_key in stat_dt_keys:
    print(dt_key)
    res_filtered = results_sims_stat.loc[dt_key]
    res_filtered['Error_X2', 'mean'].plot(ax = ax, kind='line', marker='o', 
                                            yerr=res_filtered['Error_X2', 'std'].values.T,
                                            label=dt_key, color = colors[i])
    i += 1
    
plt.xlabel('Number of Cells')
plt.ylabel('Error X2 %') 
plt.title('Simulation Mean Error for X2 when \n compared with Analytical Solution')
plt.yscale('log') 
plt.xscale('log')    
plt.legend(loc='best', title="Numerical Integration Step size")

name_plot = ('/Error_x2.png')      
plt.savefig(images_directory+name_plot, bbox_inches='tight')
#plt.show()



# Plot error vs time step 
# Error = |Value - ODE| / ODE

# Plot time iteration vs time step
