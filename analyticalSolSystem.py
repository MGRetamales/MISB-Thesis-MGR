# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 15:39:00 2022

@author: MGR
"""

import numpy as np
import matplotlib.pyplot as plt

# System:
# dx1/dt = a1 - b1 * x1
# dx2/dt = a2 + a3 * x1 - b2 * x2

# Solving analitically - PPT
# x1(t) = a1/b1 - (a1-b1*x1(0))/b1 * exp(-b1*t)
# x2(t) = 𝑥2(0) * exp(-t*b2) + exp(-t*b2) * ( (a2 + a3*a1/b1)*(exp(b2*t)-1)/b2  +  (a1-b1*x1(0))/b1 * a3 * (exp((b2-b1)*t)-1)/(b2-b1) )


def analyticalSolSystem(timeParameters, ICs, modelParameters, plots, dir_plot):
    
    # Read time parameters 
    t_init = timeParameters['t_start']
    t_end = timeParameters['t_stop']
    steps = timeParameters['steps']
    time_eval = np.linspace(t_init, t_end, steps)

    # Parameter dictionary to variables 
    a1 = modelParameters['a1']
    b1 = modelParameters['b1']
    a2 = modelParameters['a2']
    a3 = modelParameters['a3']
    b2 = modelParameters['b2']

    # ICs dictionary to variables 
    x1_0 = ICs['x1']
    x2_0 = ICs['x2']


    x1_sol = a1/b1 - (a1-b1*x1_0)/b1 * np.exp(-b1*time_eval)
    x2_sol = x2_0 * np.exp(-time_eval*b2) + np.exp(-time_eval*b2) * ( (a2 + a3*a1/b1)*(np.exp(b2*time_eval)-1)/b2  - (a1-b1*x1_0)/b1 * a3 * (np.exp((b2-b1)*time_eval)-1)/(b2-b1) )

    if plots:
        fig = plt.Figure()
        plt.plot(time_eval, x1_sol, 'r')
        plt.plot(time_eval, x2_sol, 'b')
        
        plt.xlabel('Time')
        plt.ylabel("Number of mRNA molecules")
        plt.legend(['X1', 'X2'])
        
        plt.title('ODE Analytical solution')

        name_plot = ('/AnaSol_a1_'+str(a1)+'_b1_'+str(b1)+'_a2_'
                      +str(a2)+'_a3_'+str(a3)+'_b2_'+str(b2)+'_ICX1_'+str(x1_0)
                      +'_ICX2_'+str(x1_0)+'.png')
        plt.savefig(dir_plot+name_plot)
        fig.clear()  

    return time_eval, x1_sol, x2_sol


if __name__ == '__main__':
    # Then plotting analytical solution
    modelParameters = {'a1': 1, 'b1': 0.05, 'a2': 0.1, 'a3': 0.01, 'b2': 0.1}
    ICs = {'x1': 10, 'x2': 10}
    timeParameters = {'t_start':0, 't_stop':200, 'steps':10001}
    plot_dir = 'C:/Users/Carlos/Desktop/Thesis/Results/0_Methods_SyntheticData/EvalSimulations'
    
    time_eval, x1_sol, x2_sol = analyticalSolSystem(timeParameters, ICs, modelParameters, True, plot_dir)





















