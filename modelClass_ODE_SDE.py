# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 09:47:07 2022

@author: MGR

ODE/SDE models
Implementation of solve_ivp for ODE
Implementation of Euler-Maruyama for SDE


Model:
    
0 -- a1 --> x1
x1 -- b1*x1 --> 0
0 -- a2  --> x2
0 -- a3*x1 --> x2
x2 -- b2*x2 --> 0

"""
# 0. Libraries and directories
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt



###############################################################################
# ODEs/SDEs 
###############################################################################


# 1 ODEs

# 1.1 Generate model
def ODE_model(t, x, a1, b1, a2, a3, b2):
    x1, x2 = x
    a = np.array([a1, b1*x1, a2, a3*x1, b2*x2])
    v1 = np.array([1, -1, 0, 0, 0])
    v2 = np.array([0, 0, 1, 1, -1])
    
    dx1 = sum(a*v1)
    dx2 = sum(a*v2)

    return [dx1, dx2]

# 1.2 Integrate with solve_ivp and plot
def runODE_model(timeParameters, number_cells, ICs, modelParameters, plots, dir_plot, solver):
    time_ODE = []
    results_ODE = []
    
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
    
    if plots:
        fig = plt.Figure()
    
    # Run Solve_ivp for the number of cells and save the results
    for ncell in range(number_cells):
        solODE = solve_ivp(ODE_model, [t_init, t_end], [x1_0, x2_0], args=(a1, b1, a2, a3, b2), t_eval = time_eval, method = solver)
        time = solODE.t
        time_ODE.append(time)
        results_ODE.append(solODE.y)
        
        # Plot the initial 10 to have an idea
        if plots:
            if ncell < 10:
                x1_t = solODE.y[0]
                x2_t = solODE.y[1]
                plt.plot(time, x1_t, 'r', time, x2_t, 'b')

    if plots:
        plt.xlabel('Time')
        plt.ylabel('Number of mRNA molecules')
        plt.title("ODE Numerical Approximation")
        plt.legend(['X1', 'X2'], loc=1)
    
        name_plot = ('/ODE_a1_'+str(a1)+'_b1_'+str(b1)+'_a2_'+str(a2)+'_a3_'
                     +str(a3)+'_b2_'+str(b2)+'_ICX1_'+str(x1_0)+'_ICX2_'+str(x1_0)+'.png')
        plt.savefig(dir_plot+name_plot)
        plt.show()
        fig.clear()
        plt.close()
    
    return time_ODE, results_ODE


# 2 SDEs

# 2.1.1 Use the same model that for ODE, but add noise to each ODE.

# 2.1.2 Noise is a random variable from a normal distribution mean=0, stdev = sqrt(step)
def dW(delta_t):
    """Sample a random number at each call."""
    return np.random.normal(loc=0.0, scale=np.sqrt(delta_t))

def dW_matrix(n_eq, timeParameters, n_cells):
    
    t_init = timeParameters['t_start']
    t_end = timeParameters['t_stop']
    steps = timeParameters['steps']
    dt = float(t_end - t_init) / steps
    ts = np.arange(t_init, t_end + dt, dt)
    
    noise = np.random.normal(loc=0.0, scale=np.sqrt(dt), size=[n_eq, n_cells, ts.size+1])
    return noise
    

    
# 2.1.3 Amplitude of the noise is estimated with langevin equation
def amplitude(t, x, a1, b1, a2, a3, b2, delta_t):
    x1, x2 = x
    a_sq = np.sqrt(np.array([a1, b1*x1, a2, a3*x1, b2*x2]))
    v1 = np.array([1, -1, 0, 0, 0])
    v2 = np.array([0, 0, 1, 1, -1])
    
    #Call noise function - one for each reaction
    dW_it = [dW(delta_t) for a in range(len(a_sq))] 
    
    am1 = sum(a_sq*v1*dW_it)
    am2 = sum(a_sq*v2*dW_it)
    
    return [am1, am2]

# 2.1.3 Amplitude of the noise is estimated with langevin equation
def amplitude_initializedNoise(t, x, a1, b1, a2, a3, b2, dW_it):
    x1, x2 = x
    a_sq = np.sqrt(np.array([a1, b1*x1, a2, a3*x1, b2*x2]))
    v1 = np.array([1, -1, 0, 0, 0])
    v2 = np.array([0, 0, 1, 1, -1])
    
    #Call noise function - one for each reaction
    
    am1 = sum(a_sq*v1*dW_it)
    am2 = sum(a_sq*v2*dW_it)
    
    return [am1, am2]


# 2.2 Function to run N simulations of Euler Maruyama algorithm as number of cells required
def runEulerMaruyama(timeParameters, number_cells, ICs, modelParameters, dir_plot):
    
    # Read time parameters 
    t_init = timeParameters['t_start']
    t_end = timeParameters['t_stop']
    steps = timeParameters['steps']
    
    # Read model parameters 
    a1 = modelParameters['a1']
    b1 = modelParameters['b1']
    a2 = modelParameters['a2']
    a3 = modelParameters['a3']
    b2 = modelParameters['b2']
    
    # Rad ICs
    x1_0 = ICs['x1']
    x2_0 = ICs['x2']
    
    # To store all solutions
    x_solutions = []
    
    # step size
    dt = float(t_end - t_init) / steps
    
    # Times integrated
    ts = np.arange(t_init, t_end + dt, dt)
    
    fig = plt.Figure()
    
    for ncell in range(number_cells):
    
        # Create array to store values
        xs = np.zeros([steps + 1,2])
        # Set ICs
        xs[0,:] = [x1_0, x2_0]
        
        for time_step in range(1, ts.size):
            
            t = t_init + (time_step - 1) * dt
            x = xs[time_step - 1,:]
            
            # Call ODE
            step =  np.array(ODE_model(t, x, a1, b1, a2, a3, b2))*dt
            # Call Noise
            noise = np.array(amplitude(t, x, a1, b1, a2, a3, b2, dt))
            
            # Add them to x, to create x+1
            xs_it = x + step + noise
            
            # Truncate all the variables that would go to 0 in this iteration, set them at 0
            for x_it in range(len(xs_it)):
                if xs_it[x_it] >= 0:
                    xs[time_step, x_it] = xs_it[x_it]
                else: 
                    xs[time_step, x_it] = 0
            
        # # plot the first 10 simulations to have a graphical idea
        if ncell < 5:    
            plt.plot(ts, xs[:,0], 'r')
            plt.plot(ts, xs[:,1], 'b')
        
        # print(xs[-1,:])
        
        #Save solutions
        x_solutions.append(xs)
        
        
    plt.xlabel('Time')
    plt.title('SDE Numerical Simulation')
    plt.ylabel("Number of mRNA molecules")
    plt.legend(['X1', 'X2'], loc=1)

    name_plot = ('/SDE_N_'+str(steps)+'_a1_'+str(a1)+'_b1_'+str(b1)+'_a2_'
                  +str(a2)+'_a3_'+str(a3)+'_b2_'+str(b2)+'_ICX1_'+str(x1_0)
                  +'_ICX2_'+str(x1_0)+'.png')
    plt.savefig(dir_plot+name_plot)
    plt.show()
    fig.clear()  
    plt.close()

    return ts, x_solutions


def runEulerMaruyama_noplot(timeParameters, number_cells, ICs, modelParameters):
    
    # Read time parameters 
    t_init = timeParameters['t_start']
    t_end = timeParameters['t_stop']
    steps = timeParameters['steps']
    
    # Read model parameters 
    a1 = modelParameters['a1']
    b1 = modelParameters['b1']
    a2 = modelParameters['a2']
    a3 = modelParameters['a3']
    b2 = modelParameters['b2']
    
    # Rad ICs
    x1_0 = ICs['x1']
    x2_0 = ICs['x2']
    
    # To store all solutions
    x_solutions = []
    
    # step size
    dt = float(t_end - t_init) / steps
    
    # Times integrated
    ts = np.arange(t_init, t_end + dt, dt)
    
    
    for ncell in range(number_cells):
    
        # Create array to store values
        xs = np.zeros([steps + 1,2])
        # Set ICs
        xs[0,:] = [x1_0, x2_0]
        
        for time_step in range(1, ts.size):
            
            t = t_init + (time_step - 1) * dt
            x = xs[time_step - 1,:]
            
            # Call ODE
            step =  np.array(ODE_model(t, x, a1, b1, a2, a3, b2))*dt
            # Call Noise
            noise = np.array(amplitude(t, x, a1, b1, a2, a3, b2, dt))
            
            # Add them to x, to create x+1
            xs_it = x + step + noise
            
            # Truncate all the variables that would go to 0 in this iteration, set them at 0
            for x_it in range(len(xs_it)):
                if xs_it[x_it] >= 0:
                    xs[time_step, x_it] = xs_it[x_it]
                else: 
                    xs[time_step, x_it] = 0
            
        
        
        # print(xs[-1,:])
        
        #Save solutions
        x_solutions.append(xs)
        

    return ts, x_solutions

# 2.2 Function to run N simulations of Euler Maruyama algorithm as number of cells required
# Using a fixed noise matrix predetermined
def runEulerMaruyama_initializedNoise(timeParameters, number_cells, ICs, modelParameters, noise_matrix):
    
    # Read time parameters 
    t_init = timeParameters['t_start']
    t_end = timeParameters['t_stop']
    steps = timeParameters['steps']
    
    # Read model parameters 
    a1 = modelParameters['a1']
    b1 = modelParameters['b1']
    a2 = modelParameters['a2']
    a3 = modelParameters['a3']
    b2 = modelParameters['b2']
    
    # Rad ICs
    x1_0 = ICs['x1']
    x2_0 = ICs['x2']
    
    # To store all solutions
    x_solutions = []
    
    # step size
    dt = float(t_end - t_init) / steps
    
    # Times integrated
    ts = np.arange(t_init, t_end + dt, dt)
    
    for ncell in range(number_cells):
    
        # Create array to store values
        xs = np.zeros([steps + 1,2])
        # Set ICs
        xs[0,:] = [x1_0, x2_0]
        
        for time_step in range(1, ts.size):
            
            t = t_init + (time_step - 1) * dt
            x = xs[time_step - 1,:]
            
            # Call ODE
            step =  np.array(ODE_model(t, x, a1, b1, a2, a3, b2))*dt
            # Call Noise
            dW_it = noise_matrix[:,ncell,time_step]
            noise = np.array(amplitude_initializedNoise(t, x, a1, b1, a2, a3, b2, dW_it))
            
            # Add them to x, to create x+1
            xs_it = x + step + noise
            
            # Truncate all the variables that would go to 0 in this iteration, set them at 0
            for x_it in range(len(xs_it)):
                if xs_it[x_it] >= 0:
                    xs[time_step, x_it] = xs_it[x_it]
                else: 
                    xs[time_step, x_it] = 0
            
        x_solutions.append(xs)

    return ts, x_solutions

# #########################################################
# # To Run the simulations
# #########################################################

if __name__ == '__main__':
    modelParameters = {'a1': 1, 'b1': 0.05, 'a2': 0.1, 'a3': 0.01, 'b2': 0.1}
    ICs = {'x1': 10, 'x2': 10}
    timeParameters = {'t_start':0, 't_stop':200, 'steps':200}
    
    num_cells = 10
    
    plotsDirectory = 'C:/Users/Carlos/Desktop/Thesis/Results/0_MEthods_SyntheticData/ODE_SDE'
    
    
    time_ODE, result_ODE = runODE_model(timeParameters, 1, ICs, modelParameters, True, plotsDirectory, 'LSODA')
    time_SDE, result_SDE = runEulerMaruyama(timeParameters, num_cells, ICs, modelParameters, plotsDirectory)
    
    noise_mat = dW_matrix(5, timeParameters, num_cells)
    time_SDE_in, result_SDE_in = runEulerMaruyama_initializedNoise(timeParameters, num_cells, ICs, modelParameters, noise_mat)
