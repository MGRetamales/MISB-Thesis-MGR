# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 14:31:48 2022

@author: MGR 

Simulate with CME, Gillespie and ODE the new model

Model:
    
0 -- a1 --> x1
x1 -- b1*x1 --> 0
0 -- a2  --> x2
0 -- a3*x1 --> x2
x2 -- b2*x2 --> 0


initial conditions random for x1(0) and x2(0)
a1 and b1 are known and we are looking for a2, b2 and b3

"""


# 0. Libraries and directories
import numpy as np
import matplotlib.pyplot as plt
import gillespy2
import imageio
import os 
from time import process_time
import pandas as pd

from modelClass_ODE_SDE import runODE_model, runEulerMaruyama
from modelClass_CME import simulation_ODE_CME

#Plots config sizes:
plt.rc('font', size=12)          # controls default text sizes
plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.rc('legend', fontsize=14)    # legend fontsize
plt.rc('figure', titlesize=20)  # fontsize of the figure title


working_dir = 'C:/Users/Carlos/Desktop/Thesis'
#working_dir = 'C:/Users/Carlos/Desktop/Thesis/0_pipeline'
images_directory = '/Results'
os.chdir(working_dir)





working_dir = 'C:/Users/Carlos/Desktop/Thesis'
images_directory = '/Results/0_Methods_SyntheticData'
os.chdir(working_dir)


images_ODEs_SDEs = working_dir+images_directory+'/ODE_SDE'
images_Gillespie = working_dir+images_directory+'/Gillespie'
images_CME = working_dir+images_directory+'/CME'


if not os.path.isdir(images_ODEs_SDEs):
    os.makedirs(images_ODEs_SDEs)

if not os.path.isdir(images_Gillespie):
    os.makedirs(images_Gillespie)

if not os.path.isdir(images_CME):
    os.makedirs(images_CME)

###############################################################################
# 1. Initial Parameters
###############################################################################
modelParameters = {'a1': 1, 'b1': 0.05, 'a2': 0.1, 'a3': 0.01, 'b2': 0.1}
ICs = {'x1': 10, 'x2': 10}

n_cells = 10


## Parameter dictionary to variables 
a1 = modelParameters['a1']
b1 = modelParameters['b1']
a2 = modelParameters['a2']
a3 = modelParameters['a3']
b2 = modelParameters['b2']

## ICs dictionary to variables 
x1_0 = ICs['x1']
x2_0 = ICs['x2']

## All parameters to 1 dictionary
parametersAndICs = {'modelParameters': modelParameters, 'ICs': ICs}

# Time parameters
Time_init = 0
Time_stop = 200
steps = 201

timeParameters = {'t_start':Time_init, 't_stop':Time_stop, 'steps': steps, 'delta_t': (Time_stop-Time_init)/steps}



N_truncate = 40

name_for_plots = '_a1_'+str(a1)+'_b1_'+str(b1)+'_a2_'+str(a2)+'_a3_'+str(a3)+'_b2_'+str(b1)+'_ICX1_'+str(x1_0)+'_ICX2_'+str(x1_0)


# For snapshots:
t1 = 10
t2 = 20
t3 = 30

# Number of simulations to run
n_time_snaps = 3
N_cells_killed_per_snapshot = 1000
num_cells = n_time_snaps*N_cells_killed_per_snapshot


###############################################################################
# 2.  ODEs/SDEs 
###############################################################################



# 2.1 Run ODE model
start = process_time() 
time_ODE, result_ODE = runODE_model(timeParameters, 1, ICs, modelParameters, True, images_ODEs_SDEs, 'Radau')
end = process_time()
timeElapsed_ODE = end - start

# 2.2 Run SDE model for different number of steps - to compare results with ODE
#stepsToTry = [100, 500, 1000, 2500, 5000, 7500, 10000]
stepsToTry = [201]
#stepsToTry = [100, 500, 1000]
timeElapsed_SDE_steps = []
results_SDE = []

for step in stepsToTry:
    start = process_time()
    
    timeParametersToTry = {'t_start':0, 't_stop':200, 'steps':step}
    time_SDE_it, result_SDE_it = runEulerMaruyama(timeParametersToTry, num_cells, ICs, modelParameters, images_ODEs_SDEs)
    results_SDE.append([time_SDE_it, result_SDE_it])
    end = process_time()
    timeElapsed_SDE = end - start
    timeElapsed_SDE_steps.append(timeElapsed_SDE)



# # Compare results and errors in 3 time points:
# # For snapshots:

# #ODE is exactly the same for each trajectory, so we can use only one.
# ODE_t1 = result_ODE[0][:,t1]
# ODE_t2 = result_ODE[0][:,t2]
# ODE_t3 = result_ODE[0][:,t3]

# SDE_results_stats = np.zeros([len(stepsToTry),n_time_snaps*3*2])

# #SDE we will particionate the samples in 3 timepoints and compare the error agains ODE
# i = 0
# for result in results_SDE:
    
#     SDE_t = result[0]
    
#     SDE_t1 = result[1][:N_cells_killed_per_snapshot]
#     SDE_t1_x = np.array([SDE_t1[i][np.where(SDE_t == t1),:] for i in range(N_cells_killed_per_snapshot)])[:,0,0,:]
#     SDE_t1_mean = np.mean(SDE_t1_x, axis=0)
#     SDE_t1_std = np.std(SDE_t1_x, axis=0)
    
#     SDE_results_stats[i,0] = SDE_t1_mean[0]
#     SDE_results_stats[i,1] = SDE_t1_std[0]
#     SDE_results_stats[i,2] = abs(SDE_t1_mean[0]-ODE_t1[0])/ODE_t1[0] 
    
#     SDE_results_stats[i,3] = SDE_t1_mean[1]
#     SDE_results_stats[i,4] = SDE_t1_std[1]
#     SDE_results_stats[i,5] = abs(SDE_t1_mean[1]-ODE_t1[1])/ODE_t1[1]

#     SDE_t2 = result[1][N_cells_killed_per_snapshot:N_cells_killed_per_snapshot*2]
#     SDE_t2_x = np.array([SDE_t2[i][np.where(SDE_t == t2),:] for i in range(N_cells_killed_per_snapshot)])[:,0,0,:]
#     SDE_t2_mean = np.mean(SDE_t2_x, axis=0)
#     SDE_t2_std = np.std(SDE_t2_x, axis=0)
    
#     SDE_results_stats[i,6] = SDE_t2_mean[0]
#     SDE_results_stats[i,7] = SDE_t2_std[0]
#     SDE_results_stats[i,8] = abs(SDE_t2_mean[0]-ODE_t2[0])/ODE_t2[0]
    
#     SDE_results_stats[i,9] = SDE_t2_mean[1]
#     SDE_results_stats[i,10] = SDE_t2_std[1]
#     SDE_results_stats[i,11] = abs(SDE_t2_mean[1]-ODE_t2[1])/ODE_t2[1]
    
#     SDE_t3 = result[1][N_cells_killed_per_snapshot*2:]
#     SDE_t3_x = np.array([SDE_t3[i][np.where(SDE_t == t3),:] for i in range(N_cells_killed_per_snapshot)])[:,0,0,:]
#     SDE_t3_mean = np.mean(SDE_t3_x, axis=0)
#     SDE_t3_std = np.std(SDE_t3_x, axis=0)
    
#     SDE_results_stats[i,12] = SDE_t3_mean[0]
#     SDE_results_stats[i,13] = SDE_t3_std[0]
#     SDE_results_stats[i,14] = abs(SDE_t3_mean[0]-ODE_t3[0])/ODE_t3[0]
    
#     SDE_results_stats[i,15] = SDE_t3_mean[1]
#     SDE_results_stats[i,16] = SDE_t3_std[1]
#     SDE_results_stats[i,17] = abs(SDE_t3_mean[1]-ODE_t3[1])/ODE_t3[1]
    
#     fig, ax = plt.subplots(1, 2)
#     bp = ax[0].boxplot([SDE_t1_x[:,0],SDE_t2_x[:,0], SDE_t3_x[:,0] ])
#     means = [ODE_t1[0], ODE_t2[0], ODE_t3[0]]
#     ax[0].plot([1, 2, 3], means, 'rs')
#     ax[0].set_title('X1, steps: '+str(stepsToTry[i]))
#     ax[0].set_xticks([1, 2, 3], [t1, t2, t3])
#     ax[0].set_xlabel('Time point')
    
#     bp = ax[1].boxplot([SDE_t1_x[:,1],SDE_t2_x[:,1], SDE_t3_x[:,1] ])
#     means = [ODE_t1[1], ODE_t2[1], ODE_t3[1]]
#     ax[1].plot([1, 2, 3], means, 'rs')
#     ax[1].set_title('X2, steps: '+str(stepsToTry[i]))
#     ax[1].set_xticks([1, 2, 3], [t1, t2, t3])
#     ax[1].set_xlabel('Time point')
    
#     name_plot = ('/Boxplot_SDE_N_'+str(stepsToTry[i])+'_a1_'+str(a1)+'_b1_'+str(b1)+'_a2_'
#                  +str(a2)+'_a3_'+str(a3)+'_b2_'+str(b1)+'_ICX1_'+str(x1_0)
#                  +'_ICX2_'+str(x1_0)+'.png')
#     plt.savefig(images_ODEs_SDEs+name_plot)
    
#     i += 1
    
# df = pd.DataFrame(SDE_results_stats, columns=['T1_Mean_x1', 'T1_stdev_x1', 'T1_error_x1', 'T1_Mean_x2', 'T1_stdev_x2', 'T1_error_x2',
#                                               'T2_Mean_x1', 'T2_stdev_x1', 'T2_error_x1','T2_Mean_x2', 'T2_stdev_x2', 'T2_error_x2',
#                                               'T3_Mean_x1', 'T3_stdev_x1', 'T3_error_x1', 'T3_Mean_x2', 'T3_stdev_x2', 'T3_error_x2']) 

# df.to_excel(images_ODEs_SDEs+'/ODE_SDE_N_Comparison_100to1000.xlsx')  
###############################################################################
# 3. Gillespie
###############################################################################

# 3.1 Generate model
class Molecules2(gillespy2.Model):
    def __init__(self, parameter_values=None):
        # First call the gillespy2.Model initializer.
        gillespy2.Model.__init__(self, name='Model with 2 molecules')

        # Define parameters for the rates of creation and dissociation.
        
        a1 = gillespy2.Parameter(name='a1', expression=parametersAndICs['modelParameters']['a1'])
        b1 = gillespy2.Parameter(name='b1', expression=parametersAndICs['modelParameters']['b1'])
        a2 = gillespy2.Parameter(name='a2', expression=parametersAndICs['modelParameters']['a2'])
        a3 = gillespy2.Parameter(name='a3', expression=parametersAndICs['modelParameters']['a3'])
        b2 = gillespy2.Parameter(name='b2', expression=parametersAndICs['modelParameters']['b2'])
        
        self.add_parameter([a1, b1, a2, a3, b2])

        # Define variables for the molecular species representing M and D.
        x1 = gillespy2.Species(name='x1', initial_value=parametersAndICs['ICs']['x1'])
        x2 = gillespy2.Species(name='x2',   initial_value=parametersAndICs['ICs']['x2'])
        self.add_species([x1, x2])

        # The list of reactants and products for a Reaction object are each a
        # Python dictionary in which the dictionary keys are Species objects
        # and the values are stoichiometries of the species in the reaction.
        r_c1 = gillespy2.Reaction(name="r_creation_x1", reactants={}, products={x1:1}, propensity_function="a1")
        r_d1 = gillespy2.Reaction(name="r_degradation_x1", reactants={x1:1}, products={}, propensity_function="b1*x1")
        r_c2 = gillespy2.Reaction(name="r_creation_x2", reactants={}, products={x2:1}, propensity_function="a2")
        r_c3 = gillespy2.Reaction(name="r_creation_x2_activated_by_x1", reactants={}, products={x2:1}, propensity_function="a3*x1")
        r_d2 = gillespy2.Reaction(name="r_degradation_x2", reactants={x2:1}, products={}, propensity_function="b2*x2")
        self.add_reaction([r_c1, r_d1, r_c2, r_c3, r_d2])

        # Set the timespan for the simulation.
        self.timespan(np.linspace(Time_init, Time_stop, steps))

start = process_time()        
# 3.2 Run and Plot
modelGillespie = Molecules2(parametersAndICs)
results_Gillespie = modelGillespie.run(number_of_trajectories=num_cells, seed = 10)
end = process_time()
timeElapsed_Gill_lib = end - start

f = plt.figure()
plt.title("Gillespie Simulation")
plt.xlabel("Time")
plt.ylabel('Number of mRNA molecules')
for trajectory in range(5):
    plt.plot(results_Gillespie[trajectory]['time'],results_Gillespie.data[trajectory]['x1'],'r')
    plt.plot(results_Gillespie[trajectory]['time'],results_Gillespie.data[trajectory]['x2'],'b')
plt.legend(['X1', 'X2'], loc=1)
#plt.show()
plt.savefig(images_Gillespie+'/Gillespie'+name_for_plots+'.png')
f.clear()


# # Coded Gillespie

# sim = Time_stop*10
# x = np.zeros([2, sim+1]) # Verctor of molecule numbers [x1, x2]
# t = [0]

# x[0,0] = x1_0
# x[1,0] = x2_0


# for i in range(sim):
    
    
#     x1 = x[0,i]
#     x2 = x[1,i]
    
#     w = [a1, b1*x1, a2, a3*x1, b2*x2]
    
    
#     v = [[1, 0], [-1, 0], [0, 1], [0, 1], [0, -1]]
  
#     # Decide time for next reaction
#     total_rate = sum(w)
#     u_time = np.random.rand()
#     T = -np.log(1-u_time) / total_rate
#     t.append(T+t[i])
    
#     # Decide next reaction
    
    
#     probabilities = [i/total_rate for i in w]

#     u_react = np.random.rand()
    
#     acc_prob = 0
#     count = 0
#     for prob in probabilities:
#         acc_prob += prob
#         if u_react < acc_prob:
#             react_num = count
#             x_it = x[:,i]+v[react_num]
#         else:
#             count += 1
              
#     x[:,i+1] = x_it
    

# f = plt.figure()
# plt.title("Gillespie coded")
# plt.xlabel("Time")
# plt.ylabel('Number of mRNA molecules')
# plt.plot(t, x[0,:], 'r', t, x[1,:], 'b')
# plt.legend(['X1', 'X2'])
# #plt.show()
# plt.savefig(images_Gillespie+'/Code'+name_for_plots+'.png')
# f.clear()  


###############################################################################
# 4. CME 
###############################################################################


start = process_time()
# 1.  Create an object of the simulation class
MySimulation = simulation_ODE_CME(modelParameters, timeParameters , ICs, "MySimuName", N_truncate)

# 2.  Run a simulation of the model
MySimulation.SimulateModelODEs()

end = process_time()
timeElapsed_CME = end - start

# 3. Verify that the sum of probabilities P(0)+P(1)+P(2)+...+P(N)=1. 
# If not, it means we have truncated CME at a too small N
SumPxsOverTime = []
for index in range(len(MySimulation.VariablesToPlot[0,:])):
    SumPxsOverTime.append(sum(MySimulation.VariablesToPlot[:,index]))

# Plot sum of probabilities    
fig = plt.figure()   
plt.plot(MySimulation.time,SumPxsOverTime)
plt.xlabel('Time')
plt.ylabel(r'$\Sigma_{x1=0}^{40} \Sigma_{x2=0}^{40} P\left( x1,x2 \right)$')
plt.ylim(0, 1.2)
plt.show()
fig.savefig(images_CME+'/Probabilites_sum'+name_for_plots+'.png')
fig.clear()  



# If the Sum of Probabilities from the master equation solution is ever < 1, 
# then raise an ERROR because it means we have truncated the CME too early.
if any(P <= 0.999999 for P in SumPxsOverTime):
    print("ERROR HERE!!! Sum of Ps from CME < 1!!!!!!!")
    print("CME have been truncated too early!!!!!!!!!!")
    

# x1, x2, t as a 3D matrix

Sim_VariablesToPlot_3D = np.zeros(((N_truncate+1,N_truncate+1,MySimulation.VariablesToPlot.shape[1] )))
for time in range(MySimulation.VariablesToPlot.shape[1]):
    for N_x1 in range(N_truncate+1):
        for N_x2 in range(N_truncate+1):
            Sim_VariablesToPlot_3D[N_x1,N_x2,time] = MySimulation.VariablesToPlot[(N_truncate+1)*N_x1+N_x2,time]




for x1 in range(N_truncate):

    # 7. Plot the solutions of numerically integrating the set of ODEs constituting the CME
    fig = plt.figure()  
    for x2 in range(N_truncate+1):
        if x2 < 10:
            LineStyle = '-'
        elif x2 >= 10 and x2 < 20:
            LineStyle = ':'
        elif x2 >= 20 and x2 < 30:
            LineStyle = '--'
        else:
            LineStyle = '-.'
        plt.plot(MySimulation.time,Sim_VariablesToPlot_3D[x1, x2,:],label=r'$P( x=$' + str(x2) + '$)$',linestyle=LineStyle)
    plt.xlabel('Time')
    plt.ylabel(r'$P\left( x \right)$')
    plt.legend(loc='best')
    plt.legend( bbox_to_anchor=(1,1), ncol=3)
    plt.show()
    fig.savefig(images_CME+'/Probabilites_X2_x1_'+str(x1)+'_'+name_for_plots+'.png', bbox_inches='tight')
    
    
    # 8. Plot the same, but as a probability density over time map
    figX, ax = plt.subplots(facecolor="white")
    heatmap = ax.pcolor(Sim_VariablesToPlot_3D[x1, :,:], cmap=plt.cm.rainbow)
    cbar = figX.colorbar(heatmap)
    cbar.set_label(r'$P\left(x,t\right)$', rotation=90)
    ax.set_xlabel(r"time (in integration steps of $dt=0.01$)")
    ax.set_ylabel("Number of mRNA Molecules for Gene x2 when x1 = "+str(x1))
    ax.set_title("Numerical Solution of CME")
    
    plt.show()
    figX.savefig(images_CME+'/Heatmap_X2_x1_'+str(x1)+'_'+name_for_plots+'.png')
    


# Produce a GIF of how the probability x1, x2 changes with time
snapShots = [i for i in range(10)]
snapShots.extend([i*10 for i in range(1,10)])
snapShots.extend([i*100 for i in range(1,3)])
for snapShot in snapShots:
    # 8. Plot the same, but as a probability density over time map
    figX, ax = plt.subplots(facecolor="white")
    
    heatmap = ax.pcolor(Sim_VariablesToPlot_3D[:, :, snapShot], cmap=plt.cm.rainbow, vmin = 0, vmax = 0.04)
    #heatmap = ax.pcolor(Sim_VariablesToPlot_3D[:, :, snapShot], cmap=plt.cm.rainbow)
    cbar = figX.colorbar(heatmap)
    cbar.set_label(r'$P\left(x1,x2,t\right)$', rotation=90)
    ax.set_xlabel("Number of mRNA Molecules for Gene x2")
    ax.set_ylabel("Number of mRNA Molecules for Gene x1")
    ax.set_title("Numerical Solution of CME, Snaptime = "+str(snapShot))
    
    figX.savefig(images_CME+'/Heatmap_time_'+str(snapShot)+'_'+name_for_plots+'.png')
    plt.close()
    

with imageio.get_writer(images_CME+'/'+name_for_plots+'.gif', mode='I') as writer:
    for snapShot in snapShots:
        filename = images_CME+'/Heatmap_time_'+str(snapShot)+'_'+name_for_plots+'.png'
        image = imageio.imread(filename)
        writer.append_data(image)
        writer.append_data(image)
     

snapShots = [0, 3, 6, 9, 12, 15]

for snapShot in snapShots:
    # 8. Plot the same, but as a probability density over time map
    figX, ax = plt.subplots(facecolor="white")
    
    heatmap = ax.pcolor(Sim_VariablesToPlot_3D[:, :, snapShot], cmap=plt.cm.rainbow, vmin = 0, vmax = 0.04)
    
    ax.set_xlim([0,25])
    ax.set_ylim([0,25])
    ax.set_xlabel("N° of mRNA Molecules for Gene x2")
    ax.set_ylabel("N° of mRNA Molecules for Gene x1")
    ax.set_title("Time "+str(snapShot))
    
    figX.savefig(images_CME+'/NEW_Heatmap_time_'+str(snapShot)+'_'+name_for_plots+'.png')
    
    plt.close()

###############################################################################
# 5. Plot SDE, Gillespie and CME together
###############################################################################

# times

t1 = 2
t2 = 5
t3 = 10
 
# SDE 
result_SDE_1000 = results_SDE[0]

SDE_t = result_SDE_1000[0]
result_SDE_1000_x = result_SDE_1000[1]

SDE_t1 = result_SDE_1000_x[:N_cells_killed_per_snapshot]
SDE_t1_x1 = [SDE_t1[i][t1,0] for i in range(N_cells_killed_per_snapshot)]
SDE_t1_x2 = [SDE_t1[i][t1,1] for i in range(N_cells_killed_per_snapshot)]

SDE_t2 = result_SDE_1000_x[N_cells_killed_per_snapshot: N_cells_killed_per_snapshot*2]
SDE_t2_x1 = [SDE_t2[i][t2,0] for i in range(N_cells_killed_per_snapshot)]
SDE_t2_x2 = [SDE_t2[i][t2,1] for i in range(N_cells_killed_per_snapshot)]

SDE_t3 = result_SDE_1000_x[N_cells_killed_per_snapshot*2:]
SDE_t3_x1 = [SDE_t3[i][t3,0] for i in range(N_cells_killed_per_snapshot)]
SDE_t3_x2 = [SDE_t3[i][t3,1] for i in range(N_cells_killed_per_snapshot)]




# Gillespie 

Gillespie_t1 = results_Gillespie.data[:N_cells_killed_per_snapshot]
Gillespie_t1_x1 = [Gillespie_t1[i]['x1'][t1] for i in range(N_cells_killed_per_snapshot)]
Gillespie_t1_x2 = [Gillespie_t1[i]['x2'][t1] for i in range(N_cells_killed_per_snapshot)]

Gillespie_t2 = results_Gillespie.data[N_cells_killed_per_snapshot:N_cells_killed_per_snapshot*2]
Gillespie_t2_x1 = [Gillespie_t2[i]['x1'][t2] for i in range(N_cells_killed_per_snapshot)]
Gillespie_t2_x2 = [Gillespie_t2[i]['x2'][t2] for i in range(N_cells_killed_per_snapshot)]

Gillespie_t3 = results_Gillespie.data[N_cells_killed_per_snapshot*2:]
Gillespie_t3_x1 = [Gillespie_t3[i]['x1'][t3] for i in range(N_cells_killed_per_snapshot)]
Gillespie_t3_x2 = [Gillespie_t3[i]['x2'][t3] for i in range(N_cells_killed_per_snapshot)]



# CME 
dt = timeParameters['delta_t']

CME_NumericalSolution_t1 = Sim_VariablesToPlot_3D[:,:,int(t1/dt)]
CME_NumericalSolution_t1_x1 = np.sum(CME_NumericalSolution_t1, axis=1)
CME_NumericalSolution_t1_x2 = np.sum(CME_NumericalSolution_t1, axis=0)

CME_NumericalSolution_t2 = Sim_VariablesToPlot_3D[:,:,int(t2/dt)]
CME_NumericalSolution_t2_x1 = np.sum(CME_NumericalSolution_t2, axis=1)
CME_NumericalSolution_t2_x2 = np.sum(CME_NumericalSolution_t2, axis=0)

CME_NumericalSolution_t3 = Sim_VariablesToPlot_3D[:,:,int(t3/dt)]
CME_NumericalSolution_t3_x1 = np.sum(CME_NumericalSolution_t3, axis=1)
CME_NumericalSolution_t3_x2 = np.sum(CME_NumericalSolution_t3, axis=0)


X_CME_NumericSol = range(N_truncate+1)






# Plots 
# plot the snapshot distributions 


# The 3 algorithms together: 

fig = plt.figure(figsize=(15, 5))

plt.subplot(131)
ax1=plt.subplot(1,3,1)
ax1.hist(np.array(Gillespie_t1_x1),bins=np.arange(0,50,1), histtype='stepfilled', color='r', alpha=0.5, align='left')
ax1.hist(np.array(Gillespie_t1_x2),bins=np.arange(0,50,1), histtype='stepfilled', color='b', alpha=0.5, align='left')
ax1.hist(np.array(SDE_t1_x1),bins=np.arange(0,50,1), histtype='stepfilled', color='darkred', alpha=0.5, align='left')
ax1.hist(np.array(SDE_t1_x2),bins=np.arange(0,50,1), histtype='stepfilled', color='darkblue', alpha=0.5, align='left')
ax1.plot(X_CME_NumericSol, CME_NumericalSolution_t1_x1 * N_cells_killed_per_snapshot, 'r-', marker='^')
ax1.plot(X_CME_NumericSol, CME_NumericalSolution_t1_x2 * N_cells_killed_per_snapshot, 'b-', marker='^')

ax1.set(ylabel='Cells Number')
ax1.set(title='Time: '+str(t1))



ax2=plt.subplot(1,3,2)
ax2.hist(np.array(Gillespie_t2_x1),bins=np.arange(0,50,1), histtype='stepfilled', color='r', alpha=0.5, align='left')
ax2.hist(np.array(Gillespie_t2_x2),bins=np.arange(0,50,1), histtype='stepfilled', color='b', alpha=0.5, align='left')
ax2.hist(np.array(SDE_t2_x1),bins=np.arange(0,50,1), histtype='stepfilled', color='darkred', alpha=0.5, align='left')
ax2.hist(np.array(SDE_t2_x2),bins=np.arange(0,50,1), histtype='stepfilled', color='darkblue', alpha=0.5, align='left')
ax2.plot(X_CME_NumericSol, CME_NumericalSolution_t2_x1 * N_cells_killed_per_snapshot, 'r-', marker='^')
ax2.plot(X_CME_NumericSol, CME_NumericalSolution_t2_x2 * N_cells_killed_per_snapshot, 'b-', marker='^')#ax2.set_xlim([0, x_max])
ax2.set(xlabel='Number molecules Gene X1 and X2')
ax2.set(title='Time: '+str(t2))

ax3=plt.subplot(1,3,3)
ax3.hist(np.array(Gillespie_t3_x1),bins=np.arange(0,50,1), histtype='stepfilled', color='r', alpha=0.5, align='left')
ax3.hist(np.array(Gillespie_t3_x2),bins=np.arange(0,50,1), histtype='stepfilled', color='b', alpha=0.5, align='left')
ax3.hist(np.array(SDE_t3_x1),bins=np.arange(0,50,1), histtype='stepfilled', color='darkred', alpha=0.5, align='left')
ax3.hist(np.array(SDE_t3_x2),bins=np.arange(0,50,1), histtype='stepfilled', color='darkblue', alpha=0.5, align='left')
ax3.plot(X_CME_NumericSol, CME_NumericalSolution_t3_x1 * N_cells_killed_per_snapshot, 'r-', marker='^')
ax3.plot(X_CME_NumericSol, CME_NumericalSolution_t3_x2 * N_cells_killed_per_snapshot, 'b-', marker='^')
ax3.set(title='Time: '+str(t3))
ax3.legend(['X1 - Gil', 'X2 - Gil', 'X1 - SDE', 'X2 - SDE', 'X1 - CME', 'X2 - CME'], loc="upper right")



yUpLim = max(ax1.get_ylim()[1],ax2.get_ylim()[1],ax3.get_ylim()[1]) + 5
ax1.set_ylim([0, yUpLim])
ax2.set_ylim([0, yUpLim])
ax3.set_ylim([0, yUpLim])

# plt.show()
plt.savefig(working_dir+images_directory+'/SDE_Gillespie_CME_sims_snapshots_'+name_for_plots+'.png')
fig.clear()  




# SDE and CME

fig = plt.figure(figsize=(15, 5))

plt.subplot(131)
ax1=plt.subplot(1,3,1)
ax1.hist(np.array(SDE_t1_x1),bins=np.arange(0,50,1), histtype='stepfilled', color='darkred', alpha=0.5, align='left')
ax1.hist(np.array(SDE_t1_x2),bins=np.arange(0,50,1), histtype='stepfilled', color='darkblue', alpha=0.5, align='left')
ax1.plot(X_CME_NumericSol, CME_NumericalSolution_t1_x1 * N_cells_killed_per_snapshot, 'r-', marker='^')
ax1.plot(X_CME_NumericSol, CME_NumericalSolution_t1_x2 * N_cells_killed_per_snapshot, 'b-', marker='^')

ax1.set(ylabel='Cells Number')
ax1.set(title='Time: '+str(t1))



ax2=plt.subplot(1,3,2)
ax2.hist(np.array(SDE_t2_x1),bins=np.arange(0,50,1), histtype='stepfilled', color='darkred', alpha=0.5, align='left')
ax2.hist(np.array(SDE_t2_x2),bins=np.arange(0,50,1), histtype='stepfilled', color='darkblue', alpha=0.5, align='left')
ax2.plot(X_CME_NumericSol, CME_NumericalSolution_t2_x1 * N_cells_killed_per_snapshot, 'r-', marker='^')
ax2.plot(X_CME_NumericSol, CME_NumericalSolution_t2_x2 * N_cells_killed_per_snapshot, 'b-', marker='^')
ax2.set(xlabel='Number molecules Gene X1 and X2')
ax2.set(title='Time: '+str(t2))

ax3=plt.subplot(1,3,3)

ax3.hist(np.array(SDE_t3_x1),bins=np.arange(0,50,1), histtype='stepfilled', color='darkred', alpha=0.5, align='left')
ax3.hist(np.array(SDE_t3_x2),bins=np.arange(0,50,1), histtype='stepfilled', color='darkblue', alpha=0.5, align='left')
ax3.plot(X_CME_NumericSol, CME_NumericalSolution_t3_x1 * N_cells_killed_per_snapshot, 'r-', marker='^')
ax3.plot(X_CME_NumericSol, CME_NumericalSolution_t3_x2 * N_cells_killed_per_snapshot, 'b-', marker='^')
ax3.set(title='Time: '+str(t3))
ax3.legend(['X1 - SDE', 'X2 - SDE', 'X1 - CME', 'X2 - CME'], loc="upper right")



yUpLim = max(ax1.get_ylim()[1],ax2.get_ylim()[1],ax3.get_ylim()[1]) + 5
ax1.set_ylim([0, yUpLim])
ax2.set_ylim([0, yUpLim])
ax3.set_ylim([0, yUpLim])

# plt.show()
plt.savefig(working_dir+images_directory+'/SDE_CME_sims_snapshots_'+name_for_plots+'.png')
fig.clear()  



# Gillespie and CME

fig = plt.figure(figsize=(15, 5))
plt.subplot(131)
ax1=plt.subplot(1,3,1)
ax1.hist(np.array(Gillespie_t1_x1),bins=np.arange(0,50,1), histtype='stepfilled', color='r', alpha=0.5, align='left')
ax1.hist(np.array(Gillespie_t1_x2),bins=np.arange(0,50,1), histtype='stepfilled', color='b', alpha=0.5, align='left')
ax1.plot(X_CME_NumericSol, CME_NumericalSolution_t1_x1 * N_cells_killed_per_snapshot, 'r-', marker='^')
ax1.plot(X_CME_NumericSol, CME_NumericalSolution_t1_x2 * N_cells_killed_per_snapshot, 'b-', marker='^')

ax1.set(ylabel='Cells Number')
ax1.set(title='Time: '+str(t1))



ax2=plt.subplot(1,3,2)
ax2.hist(np.array(Gillespie_t2_x1),bins=np.arange(0,50,1), histtype='stepfilled', color='r', alpha=0.5, align='left')
ax2.hist(np.array(Gillespie_t2_x2),bins=np.arange(0,50,1), histtype='stepfilled', color='b', alpha=0.5, align='left')
ax2.plot(X_CME_NumericSol, CME_NumericalSolution_t2_x1 * N_cells_killed_per_snapshot, 'r-', marker='^')
ax2.plot(X_CME_NumericSol, CME_NumericalSolution_t2_x2 * N_cells_killed_per_snapshot, 'b-', marker='^')#ax2.set_xlim([0, x_max])
ax2.set(xlabel='Number molecules Gene X1 and X2')
ax2.set(title='Time: '+str(t2))

ax3=plt.subplot(1,3,3)
ax3.hist(np.array(Gillespie_t3_x1),bins=np.arange(0,50,1), histtype='stepfilled', color='r', alpha=0.5, align='left')
ax3.hist(np.array(Gillespie_t3_x2),bins=np.arange(0,50,1), histtype='stepfilled', color='b', alpha=0.5, align='left')
ax3.plot(X_CME_NumericSol, CME_NumericalSolution_t3_x1 * N_cells_killed_per_snapshot, 'r-', marker='^')
ax3.plot(X_CME_NumericSol, CME_NumericalSolution_t3_x2 * N_cells_killed_per_snapshot, 'b-', marker='^')
ax3.set(title='Time: '+str(t3))
ax3.legend(['X1 - Gil', 'X2 - Gil', 'X1 - CME', 'X2 - CME'], loc="upper right")



yUpLim = max(ax1.get_ylim()[1],ax2.get_ylim()[1],ax3.get_ylim()[1]) + 5
ax1.set_ylim([0, yUpLim])
ax2.set_ylim([0, yUpLim])
ax3.set_ylim([0, yUpLim])

# plt.show()
plt.savefig(working_dir+images_directory+'/Gillespie_CME_sims_snapshots_'+name_for_plots+'.png')
fig.clear()  






























