# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 10:24:31 2022

@author: MGR
Adapted from SM codes

CME model

Model:
    
0 -- a1 --> x1
x1 -- b1*x1 --> 0
0 -- a2  --> x2
0 -- a3*x1 --> x2
x2 -- b2*x2 --> 0

The model is truncated for 0 min number of molecules
The model is truncated for N_truncate max number of molecules

"""

# 0. Libraries and directories
import numpy as np
from scipy.integrate import ode




###############################################################################
# CME
###############################################################################


def ODEsCMEasFunction(t, P_variables, ParameterSet, N_truncate):
    # P_variables is a truncated matrix of len[P] = (N_truncate x N_truncate + N_truncate)
   
    # P_variables[(N_truncate+1)*i + j] = [i,j]
    
    # ============================================================================
    # Be aware that the above Chemical Master Equation (a set of ODEs) implement 
    # the following model (same as the one used for Gillespie simulations above):
    #      alpha1             beta1
    #   0  ----->  X1,    X1  ----->  0 
    #
    #      alpha2             alpha3*X1           beta2
    #   0  ----->  X2,    0  ----------->  X2 ,    X2  ----->  0 
    # =============================================================================
    
    ### 1) This function defines the system of ODEs as a function
    a1 = ParameterSet['a1']
    b1 = ParameterSet['b1']
    a2 = ParameterSet['a2']
    a3 = ParameterSet['a3']
    b2 = ParameterSet['b2']
    

    ODEsystem = []
    
    # 1. the ODE corresponding to the boundary condition dP(0,0)/dt
    RightHandSideODE_P_0_0 = (- (a1 + a2) * P_variables[0]     #P[0,0]
                            + b2 * P_variables[1]              #P[0,1]
                            + b1 * P_variables[N_truncate+1])  #P[1,0]
    ODEsystem.append(RightHandSideODE_P_0_0)
    
    # 2. find the ODEs for boundary  dP(0,x2)/dt
    for x2 in range(1,N_truncate):
        # x2 loop
        RightHandSideODE_P_0_x2 = (-(a1 + a2 + b2*x2) * P_variables[x2]      #P[0,x2]
                                 + a2 * P_variables[x2-1]                    #P[0,x2-1]
                                 + b2 * (x2+1) * P_variables[x2+1]           #P[0,x2+1]
                                 + b1 * P_variables[(N_truncate+1)+x2])    #P[1,x2]
        ODEsystem.append(RightHandSideODE_P_0_x2)

    # 3. find the ODEs for boundary  dP(0,N)/dt
    RightHandSideODE_P_0_N = (- (a1 + b2*N_truncate) * P_variables[N_truncate]     #P[0,N]
                            + b1 * P_variables[(N_truncate+1)+N_truncate]          #P[1,N]
                            + a2 * P_variables[N_truncate-1])                      #P[0,N-1]
    ODEsystem.append(RightHandSideODE_P_0_N)

    
    
    # 4. fill in with all the other ODEs for dP(x1, x2)/dt for X = 1, ..., N
    for x1 in range(1,N_truncate):
        # x1 loop
        
        # 4.1 find the ODEs for boundary  dP(x1,0)/dt
        RightHandSideODE_P_x1_0 = (-(a1 + b1 * x1 + a2 + a3*x1) * P_variables[(N_truncate+1)*x1] #P[x1,0]
                                 + a1 * P_variables[(N_truncate+1)*(x1-1)]                       #P[x1-1,0]
                                 + b2 * P_variables[(N_truncate+1)*x1 + 1]                       #P[x1,1]
                                 + b1 * (x1+1) * P_variables[(N_truncate+1)*(x1+1)])             #P[x1+1,0]
        ODEsystem.append(RightHandSideODE_P_x1_0)

        
        # 4.2 Loop though all posible values of x2    
        for x2 in range(1, N_truncate):

            # x2 loop
            RightHandSideODE_P_x1_x2 = (-(a1 + b1*x1 + a2 + a3*x1 + b2*x2)  * P_variables[(N_truncate+1)*x1 + x2] #P[x1,x2]
                                        + a1 * P_variables[(N_truncate+1)*(x1-1) + x2]                            #P[x1-1,x2]
                                        + b1 * (x1+1) * P_variables[(N_truncate+1)*(x1+1) + x2]                   #P[x1+1,x2]
                                        + (a2 + a3*x1) * P_variables[(N_truncate+1)*x1 + x2 - 1]                  #P[x1,x2-1]
                                        + b2 * (x2+1) * P_variables[(N_truncate+1)*x1 + x2 + 1])                  #P[x1,x2+1]

            ODEsystem.append(RightHandSideODE_P_x1_x2)

           
        # 4.3 find the ODEs for boundary  dP(x1,N)/dt
        RightHandSideODE_P_x1_N = (-(a1 + b1 * x1 + b2 * N_truncate) * P_variables[(N_truncate+1)*x1+N_truncate] #P[x1,N]
                                 + a1 * P_variables[(N_truncate+1)*(x1-1)+N_truncate]                            #P[x1-1,N]
                                 + b1 * (x1 + 1) * P_variables[(N_truncate+1)*(x1+1) + N_truncate]               #P[x1+1,N]
                                 + (a2 + a3 * x1) * P_variables[(N_truncate+1)*x1 + N_truncate - 1])             #P[x1,N-1]
        ODEsystem.append(RightHandSideODE_P_x1_N)

    
    # 5. Find last ODE dP(N,0)/dt
    RightHandSideODE_P_N_0 = (-(b1 * N_truncate + a2 + a3*N_truncate) * P_variables[(N_truncate+1)*N_truncate] #P[N,0]
                             + a1 * P_variables[(N_truncate+1)*(N_truncate-1)]                                 #P[N-1,0]
                             + b2 * P_variables[(N_truncate+1)*N_truncate + 1])                                #P[N,1]
    ODEsystem.append(RightHandSideODE_P_N_0) 

    # 6. Find last ODE dP(N,x2)/dt
    for x2 in range(1,N_truncate):
        # x2 loop
        RightHandSideODE_P_N_x2 = (-(b1*N_truncate + a2 + a3*N_truncate + b2*x2) *P_variables[(N_truncate+1)*N_truncate + x2] #P[N,x2]
                                 + (a2 + a3*N_truncate) * P_variables[ (N_truncate+1)*N_truncate + x2 - 1]                    #P[N,x2-1]
                                 + b2 * (x2+1) * P_variables[(N_truncate+1)*N_truncate + x2+1]                                #P[N,x2+1]
                                 + a1 *  P_variables[(N_truncate+1)*(N_truncate-1) + x2])                                     #P[N-1,x2]
        ODEsystem.append(RightHandSideODE_P_N_x2)


    # 7. Find last ODE dP(N,N)/dt
    RightHandSideODE_N_N = (-(b1*N_truncate + b2*N_truncate)  * P_variables[(N_truncate+1)*(N_truncate)+N_truncate] #P[N,N]
                             + a1 * P_variables[(N_truncate+1)*(N_truncate-1)+N_truncate]                           #P[N-1,N]
                             + (a2 + a3*N_truncate) * P_variables[(N_truncate+1)*(N_truncate)+N_truncate-1])        #P[N,N-1]
    
    ### POTENTIAL PROBLEM HERE: THIS WAY OF TRUNCATING ASSUMES LAST P IS 0, 
    # WHICH ONLY HOLDS IF WE ARE SOLVING SUFFICIENTLY MANY ODES, 
    # WHICH IN TURN OBLIGE US LATER TO SOLVE MORE ODEs THAT NEEDED BY MAX(DATA), i.e. by COMPUTATION OF LIKELIHOOD
    ODEsystem.append(RightHandSideODE_N_N)

    return ODEsystem

# CAN BE GENERALIZED AND KEPT FOR ALL MODELS! Script Functions B - CME for ProdDeg Model
class simulation_ODE_CME:
    """ Creates a simulation of ODEs system"""

    def __init__(self, ModelParameters, TimeParameters, ParametersSetIC, NameOfSimulation, N_truncate, IntegrationMethod = 'lsoda'):
        
        self.name = NameOfSimulation
        self.ModelParameters = ModelParameters
        self.TimeParameters = TimeParameters
        self.ParametersSetIC = ParametersSetIC
        self.N_truncate = N_truncate

        # 1. Define the desired number of time steps (+1 for initial condit).
        self.StepsNum = np.floor((self.TimeParameters["t_stop"] - self.TimeParameters["t_start"]) / self.TimeParameters["delta_t"]) + 1 
       
        # 2. Specify integrator ('lsoda' choses automatically between 'bdf' and 'Adams'):
        self.ODEs = ode(ODEsCMEasFunction).set_integrator(IntegrationMethod) # ('lsoda')  'dopri5'  'dop853'  'vode'
        
        # nsteps=1000 (maximum number of steps allowed), max_step=1.0 (maximal lenght of a step)
    
        # 3. Put initial conditions into an array
        # Remember: P_variables[(N_truncate+1)*i + j] = [i,j]
        self.position0 = np.zeros((N_truncate+1)*(N_truncate+1))
        # P_variables[(N_truncate+1)*i + j] = [i,j]
        self.position0[(N_truncate+1)*(self.ParametersSetIC['x1'])+self.ParametersSetIC['x2']] = 1
    
        # 4. Prepare arrays to contain the values of the variables for plotting
        self.time = np.empty((int(self.StepsNum), ))
        
        # Now we need to generate an array containing all the variables
        self.VariablesToPlot = np.zeros(((N_truncate+1)*(N_truncate+1),int(self.StepsNum)))

    def SimulateModelODEs(self):
        i = 0
        ### 8) Set initial conditions for the integrating variable, time parameters sets and parameters of the model
        self.ODEs.set_initial_value(self.position0, self.ODEs.t).set_f_params(self.ModelParameters,self.N_truncate)
        ### 9) Integrate the ODEs across each delta_t timestep (i.e. compute y(t) for every t = n * delta_t)
        
        while self.ODEs.successful() and i < self.StepsNum:
        #while i < self.StepsNum:
            self.ODEs.integrate(self.ODEs.t + self.TimeParameters["delta_t"])
    
            ### 10) ...and fill in the solution into the arrays for plotting
            self.time[i] = self.ODEs.t

            #for index_variable in range(self.N_truncate+1):
            self.VariablesToPlot[:,i] = self.ODEs.y
            
            i += 1

# #########################################################
# # To Run the simulations
# #########################################################


if __name__ == '__main__':
    modelParameters = {'a1': 1, 'b1': 0.05, 'a2': 0.1, 'a3': 0.01, 'b2': 0.1}
    ICs = {'x1': 10, 'x2': 10}
    # Time parameters
    Time_init = 0
    Time_stop = 1000
    steps = 1001
    timeParameters = {'t_start':Time_init, 't_stop':Time_stop, 'steps': steps, 'delta_t': (Time_stop-Time_init)/steps}
    N_truncate = 40
    
    
    MySimulation = simulation_ODE_CME(modelParameters, timeParameters , ICs, "MySimuName", N_truncate)
    MySimulation.SimulateModelODEs()


