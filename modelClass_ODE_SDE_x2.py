# -*- coding: utf-8 -*-
"""
Created on Tue May 17 20:40:31 2022


@author: MGR

ODE/SDE models
Implementation of solve_ivp for ODE
Implementation of Euler-Maruyama for SDE


Model:
    
0 -- a2  --> x2
0 -- a3*x1 --> x2
x2 -- b2*x2 --> 0

x1 expression is interpolated over time, so it is only another input

"""
# 0. Libraries and directories
import numpy as np



###############################################################################
# ODEs/SDEs 
###############################################################################


# 1 ODEs

# 1.1 Generate model
def ODE_model(t, x2, a2, a3, b2, x1):
    
    a = np.array([a2, a3*x1, b2*x2])
    v2 = np.array([1, 1, -1])
    
    dx2 = sum(a*v2)

    return [dx2]


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
def amplitude_initializedNoise_pr(t, x2, a2, a3, b2, x1, dW_it):
    a_sq = np.sqrt(np.array([a2, a3*x1, b2*x2]))
    v2 = np.array([1, 1, -1])
    
    #Call noise function - one for each reaction
    am2 = sum(a_sq*v2*dW_it)
    
    return [am2]

def amplitude_initializedNoise_nr(t, x2, a2, a3, b2, x1, dW_it):
    a_sq = np.sqrt(abs(np.array([a2, a3*x1, b2*x2])))
    v2 = np.array([1, -1, -1])
    
    #Call noise function - one for each reaction
    am2 = sum(a_sq*v2*dW_it)
    
    return [am2]


# 2.2 Function to run N simulations of Euler Maruyama algorithm as number of cells required
# Using a fixed noise matrix predetermined
def runEulerMaruyama_initializedNoise_x1interpolated(timeParameters, number_cells, ICs, modelParameters, noise_matrix, x1_expression):
    
    # Read time parameters 
    t_init = timeParameters['t_start']
    t_end = timeParameters['t_stop']
    steps = timeParameters['steps']
    
    # Read model parameters 
    a2 = modelParameters['a2']
    a3 = modelParameters['a3']
    b2 = modelParameters['b2']
    
    # Rad ICs
    x2_0 = ICs['x2']
    
    # To store all solutions
    x_solutions = []
    
    # step size
    dt = float(t_end - t_init) / steps
    
    # Times integrated
    ts = np.arange(t_init, t_end + dt, dt)
    
    for ncell in range(number_cells):
        
        x1_cell = x1_expression[:,ncell]
    
        # Create array to store values
        xs = np.zeros([steps + 1])
        # Set ICs
        xs[0] = x2_0
        
        for time_step in range(1, ts.size):
        #for time_step in range(1, 10):
            
            t = t_init + (time_step - 1) * dt
            x2 = xs[time_step - 1]
            x1 = x1_cell[time_step - 1]
            
            # Call ODE
            step =  np.array(ODE_model(t, x2, a2, a3, b2, x1))*dt
            # Call Noise
            dW_it = noise_matrix[:,ncell,time_step]
            noise = np.array(amplitude_initializedNoise_pr(t, x2, a2, a3, b2, x1, dW_it))
            
            # Add them to x, to create x+1
            xs_it = x2 + step + noise
            
            # Truncate all the variables that would go to 0 in this iteration, set them at 
            if xs_it >= 0:
                xs[time_step] = xs_it
            else: 
                xs[time_step] = 0
            
        x_solutions.append(xs)


    return ts, x_solutions


# 2.2 Function to run N simulations of Euler Maruyama algorithm as number of cells required
# Using a fixed noise matrix predetermined
def runEM_initNoise_x1int_x2ICs(timeParameters, x2_ICs, modelParameters, noise_matrix, x1_expression):
    
    # There is one IC per cell, they don't use the same value
    # It matches the cell of the x1_expression 
    number_cells = len(x2_ICs)
    
    # Read time parameters 
    t_init = timeParameters['t_start']
    t_end = timeParameters['t_stop']
    steps = timeParameters['steps']
    
    # Read model parameters 
    a2 = modelParameters['a2']
    a3 = modelParameters['a3']
    b2 = modelParameters['b2']
    
    
    # To store all solutions
    x_solutions = []
    
    # step size
    dt = float(t_end - t_init) / steps
    
    # Times integrated
    ts = np.arange(t_init, t_end + dt, dt)
    
    for ncell in range(number_cells):
        
        x1_cell = x1_expression[:,ncell]
        x2_0 = x2_ICs[ncell]
    
        # Create array to store values
        xs = np.zeros([steps + 1])
        # Set ICs
        xs[0] = x2_0
        
        for time_step in range(1, ts.size):
        #for time_step in range(1, 10):
            
            t = t_init + (time_step - 1) * dt
            x2 = xs[time_step - 1]
            x1 = x1_cell[time_step - 1]
            
            # Call ODE
            step =  np.array(ODE_model(t, x2, a2, a3, b2, x1))*dt
            # Call Noise
            dW_it = noise_matrix[:,ncell,time_step]
            
            if a3 >= 0:
                noise = np.array(amplitude_initializedNoise_pr(t, x2, a2, a3, b2, x1, dW_it))
            
            else:
                noise = np.array(amplitude_initializedNoise_nr(t, x2, a2, a3, b2, x1, dW_it))
                
            # Add them to x, to create x+1
            xs_it = x2 + step + noise
            
            # Truncate all the variables that would go to 0 in this iteration, set them at 
            if xs_it >= 0:
                xs[time_step] = xs_it
            else: 
                xs[time_step] = 0
            
        x_solutions.append(xs)


    return ts, x_solutions



# 2.2 Function to run N simulations of Euler Maruyama algorithm as number of cells required
# Using a fixed noise matrix predetermined 
# Fit x2 without x1
def runEM_initNoise_x2ICs(timeParameters, x2_ICs, modelParameters, noise_matrix):
    
    # There is one IC per cell, they don't use the same value
    # It matches the cell of the x1_expression 
    number_cells = len(x2_ICs)
    
    # Read time parameters 
    t_init = timeParameters['t_start']
    t_end = timeParameters['t_stop']
    steps = timeParameters['steps']
    
    # Read model parameters 
    a2 = modelParameters['a2']
    a3 = 0
    b2 = modelParameters['b2']
    
    
    # To store all solutions
    x_solutions = []
    
    # step size
    dt = float(t_end - t_init) / steps
    
    # Times integrated
    ts = np.arange(t_init, t_end + dt, dt)
    
    for ncell in range(number_cells):
        
        x2_0 = x2_ICs[ncell]
    
        # Create array to store values
        xs = np.zeros([steps + 1])
        # Set ICs
        xs[0] = x2_0
        
        for time_step in range(1, ts.size):
        #for time_step in range(1, 10):
            
            t = t_init + (time_step - 1) * dt
            x2 = xs[time_step - 1]
            
            # Call ODE
            step =  np.array(ODE_model(t, x2, a2, a3, b2, 0))*dt
            # Call Noise
            dW_it = noise_matrix[:,ncell,time_step]
            noise = np.array(amplitude_initializedNoise_pr(t, x2, a2, a3, b2, 0, dW_it))
            
            # Add them to x, to create x+1
            xs_it = x2 + step + noise
            
            # Truncate all the variables that would go to 0 in this iteration, set them at 
            if xs_it >= 0:
                xs[time_step] = xs_it
            else: 
                xs[time_step] = 0
            
        x_solutions.append(xs)


    return ts, x_solutions





def runEM_initNoise_x1int_x2ICs_ncell(timeParameters, x2_ICs, modelParameters, noise_matrix, x1_expression, cells_id):
    
    # There is one IC per cell, they don't use the same value
    # It matches the cell of the x1_expression 
    number_cells = len(cells_id)
    
    # Read time parameters 
    t_init = timeParameters['t_start']
    t_end = timeParameters['t_stop']
    steps = timeParameters['steps']
    
    # Read model parameters 
    a2 = modelParameters['a2']
    a3 = modelParameters['a3']
    b2 = modelParameters['b2']
    
    
    # To store all solutions
    x_solutions = []
    
    # step size
    dt = float(t_end - t_init) / steps
    
    # Times integrated
    ts = np.arange(t_init, t_end + dt, dt)
    
    for ncell in cells_id:
        
        x1_cell = x1_expression[:,ncell]
        x2_0 = x2_ICs[ncell]
    
        # Create array to store values
        xs = np.zeros([steps + 1])
        # Set ICs
        xs[0] = x2_0
        
        for time_step in range(1, ts.size):
        #for time_step in range(1, 10):
            
            t = t_init + (time_step - 1) * dt
            x2 = xs[time_step - 1]
            x1 = x1_cell[time_step - 1]
            
            # Call ODE
            step =  np.array(ODE_model(t, x2, a2, a3, b2, x1))*dt
            # Call Noise
            dW_it = noise_matrix[:,ncell,time_step]
            noise = np.array(amplitude_initializedNoise(t, x2, a2, a3, b2, x1, dW_it))
            
            # Add them to x, to create x+1
            xs_it = x2 + step + noise
            
            # Truncate all the variables that would go to 0 in this iteration, set them at 
            if xs_it >= 0:
                xs[time_step] = xs_it
            else: 
                xs[time_step] = 0
            
        x_solutions.append(xs)


    return ts, x_solutions




# #########################################################
# # To Run the simulations
# #########################################################
# modelParameters = {'a2': 0.1, 'a3': 0.01, 'b2': 0.1}
# ICs = {'x2': 10}
# timeParameters = {'t_start':0, 't_stop':81, 'steps':81}
# num_cells = 1000



# noise_mat = dW_matrix(3, timeParameters, num_cells)
# time_SDE, result_SDE = runEulerMaruyama_initializedNoise_x1interpolated(timeParameters, num_cells, ICs, modelParameters, noise_mat, x_interpolated)
# time_SDE, result_SDE = runEM_initNoise_x1int_x2ICs(timeParameters, x2_ICs, modelParameters, noise_matrix, x1_expression)