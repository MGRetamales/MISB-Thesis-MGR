# -*- coding: utf-8 -*-
"""
Created on Fri May  6 09:33:08 2022

@author: MGR

Plots created for SDEs and cost functions

1. plot_sde_data: 
    runs SDE simulation + plots the distributions with the real data for
    the timepoints of the real data. 
    Also prints the wasserstein distance in 1D for them
    
    

"""
import numpy as np 
from matplotlib import pyplot as plt



from modelClass_ODE_SDE import  runEulerMaruyama_noplot
from wasserstein_distance import wasserstein_distance_1D



from sde_cost_functions import sde_cf_notScaled_notInitNoise_l1, sde_cf_notScaled_notInitNoise_l2, sde_cf_Scaled_notInitNoise_l1, sde_cf_Scaled_notInitNoise_l2
from sde_cost_functions import sde_cf_notScaled_InitNoise_l1, sde_cf_notScaled_InitNoise_l2, sde_cf_Scaled_InitNoise_l1, sde_cf_Scaled_InitNoise_l2





#Plots config sizes:
plt.rc('font', size=12)          # controls default text sizes
plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.rc('legend', fontsize=14)    # legend fontsize
plt.rc('figure', titlesize=20)  # fontsize of the figure title
    
    
    
def plot_sde_data(ab, x0, tspan, data, num_cells, wd_param, plots_directory):
    modelParameters = {'a1': ab[0], 'b1':ab[1], 'a2':ab[2], 'a3': ab[3], 'b2': ab[4]}
    timeParameters = {'t_start':0, 't_stop':tspan[-1], 'steps':tspan[-1]}
    
    nTrunc = wd_param[1]
    plot_name = '_'.join(list(map(str, ab)))
    
    _, result_SDE_it = runEulerMaruyama_noplot(timeParameters, num_cells, x0, modelParameters)
    
    # Dstack (time, x, cell)
    result_SDE_it = np.dstack(result_SDE_it)
    
    
    for time in tspan:
        real = data[str(time)]
        simulated = np.transpose(result_SDE_it[time,:,:])
        
        fig = plt.figure(figsize=(12, 5), constrained_layout=True)
        #plt.rcParams['font.size'] = '14'
        fig.suptitle('Timepoint:  '+str(time))
        
        ax1 = plt.subplot(1,2,1)
        ax1.hist(real[:,0],bins=np.arange(0,nTrunc,1), histtype='stepfilled', color='r', alpha=0.5, align='left')
        ax1.hist(simulated[:,0],bins=np.arange(0,nTrunc,1), histtype='stepfilled', color='b', alpha=0.5, align='left')
        ax1.set(title='X1', xlabel='mRNA molecules', ylabel='N° of cells')
        ax1.set_ylim(0, 400)
        ax1.legend(['Synthetic','Estimated'])
        
        ax2 = plt.subplot(1,2,2)
        ax2.hist(real[:,1],bins=np.arange(0,nTrunc,1), histtype='stepfilled', color='r', alpha=0.5, align='left')
        ax2.hist(simulated[:,1],bins=np.arange(0,nTrunc,1), histtype='stepfilled', color='b', alpha=0.5, align='left')
        ax2.set(title='X2', xlabel='mRNA molecules', ylabel='N° of cells')
        ax2.set_ylim(0, 400)
        ax2.legend(['Synthetic','Estimated'])

    
        print(time)

        wd_x1 = wasserstein_distance_1D(real[:,0], simulated[:,0], wd_param[0], wd_param[1], wd_param[2])
        wd_x2 = wasserstein_distance_1D(real[:,1], simulated[:,1], wd_param[0], wd_param[1], wd_param[2])
        
        ax1.text(nTrunc/3,250,'WD: '+format(wd_x1, ".2g"), fontsize=20, bbox=dict(boxstyle="square", ec=(0., 0., 0.), fc=(1., 1., 1.),))
        ax2.text(nTrunc/3,250,'WD: '+format(wd_x2, ".2g"), fontsize=20, bbox=dict(boxstyle="square", ec=(0., 0., 0.), fc=(1., 1., 1.),))

        
        plt.savefig(plots_directory+'/Distance_t_'+str(time)+'_ab_'+plot_name+'.png')
        plt.show()
        fig.clear()
        
        print('WD x1: '+str(wd_x1))
        print('WD x2: '+str(wd_x2))
        
# plot_sde_data(ab_exp, x0, tspan, expression, 1000, [1, 50, 0.1], results_directory)


def plot_cf_1P(CF, CF_name, ab_opt_exp, ab_names, par_try, x0, tspan, data,  num_cells, wd_param, scaling_factors, noise_mat, plots_directory):
    
    for ab_iterated in range(len(ab_opt_exp)):
        print(ab_iterated)
        ofs = []
    
        for par in par_try:
            ab_try_it = ab_opt_exp.copy()
            ab_try_it[ab_iterated] = par
            print(ab_try_it)
            of = CF(ab_try_it, x0, tspan, data, num_cells, wd_param, scaling_factors, noise_mat)
            ofs.append(of)

        plt.plot(par_try, ofs)
        plt.xlabel('Value of Parameter')
        plt.ylabel('Objective Function')
        plt.title(ab_names[ab_iterated])
        plt.savefig(plots_directory+'/OF_plots/'+CF_name+'_'+ab_names[ab_iterated]+'_'+str(par_try[0])+'_'+str(par_try[-1])+'.png')
        plt.show()
        
        
        
# plot_cf_1P(sde_cf_notScaled_notInitNoise_l1, 'NotScaled_notInit', ab_exp, ab_names, test_4, x0, tspan, expression,  1000, [1, 50, 0.1], 0, 0, results_directory)   
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        