# -*- coding: utf-8 -*-
"""
Created on Mon May 30 11:13:51 2022

@author: MGR

Gillespie simulation to obtain raw read matrices



"""

import os 
import numpy as np
import pandas as pd
import pickle
from gillespy2 import Model, Species, Parameter, Reaction



###############################################################################
# 0. Directories to read network and save results
###############################################################################

## Important directory, has to be changed in each computer
working_dir = 'C:/Users/Carlos/Desktop/Thesis'
os.chdir(working_dir)

network_name = 'network02_01'

#networks = ['network02_01' , 'network02_02', 'network02_03', 'network02_04', 'network02_05']
# networks = [ 'network05_01', 'network05_02', 'network05_03', 'network05_04', 'network05_05',
  #          'network05_06']
networks = ['network10_01', 'network10_02', 'network10_03', 'network10_04']


for network_name in networks:
    network_directory = working_dir+'/Results/00_Whole_Pipelines/'
    #network_name = 'network05'
    #network_name = input("Enter name of network:")
    print("Name of network: "+network_name)
    
    
    network_directory = network_directory+network_name
    network_file = network_directory+'.xlsx'
    
    if not os.path.isdir(network_directory):
        os.mkdir(network_directory)
    
    result_directory = network_directory+'/0_Data'
    if not os.path.isdir(result_directory):
        os.mkdir(result_directory)
    
    figures_directory = result_directory+'/Figures/'
    if not os.path.isdir(figures_directory):
        os.mkdir(figures_directory)
    
    print('Running Gillespie simulation for network: '+ network_file)
    ###############################################################################
    # 1. Create network
    ###############################################################################
    
    print('Reading File')
    network = pd.read_excel(network_file, sheet_name = 'A_matrix', index_col = 0)
    
    regulations = {}
    basal = {}
    degradations = {}
    
    links = []
    differential_equations = {}
    
    
    a1_a2_s = pd.read_excel(network_file, sheet_name = 'Basal', index_col = 0)
    
    # Basal expression - aka a1/a2
    for row in a1_a2_s.index:
        a1_a2 = a1_a2_s.loc[row]['a1']
        differential_equations[row] = str(a1_a2)
        
        if a1_a2 != 0:
    
            reg_key = 'a'+str(row[1:])
            basal[reg_key] = a1_a2.item()
            
            
            
            
    # Columns regulate rows
    for column in network.columns:
        network_col = network[column]
    
        for row in network_col.index:
            network_col_row = network_col[row]
    
            if column == row:
                deg_key = 'b'+str(column[1:])
                degradations[deg_key] = abs(network_col_row).item()
                differential_equations[row] = differential_equations[row] + ' - ' + str(abs(network_col_row)) + ' * ' + column
            
            elif network_col_row != 0:
    
                reg_key = 'a'+str(column[1:])+'_'+str(row[1:])
                regulations[reg_key] = network_col_row.item()
                links.append([column, row, 1])
                differential_equations[row] = differential_equations[row] + ' + ' + str(network_col_row) + ' * ' + column
                
                
            else:
                links.append([column, row, 0])
        
    
    
           
    
    ICs = {}
    initial_conditions = pd.read_excel(network_file, sheet_name = 'InitialConditions', index_col = 0)
    
    genes = list(initial_conditions.index)
    
    for gene in initial_conditions.index:
        ICs[gene] = initial_conditions.loc[gene]['ic'].item()
    
    
    
    ## All parameters to 1 dictionary
    parametersAndICs = {'modelParameters_act': regulations, 'modelParameters_bas': basal, 'modelParameters_deg': degradations, 'ICs': ICs}
    
    
    
    ###############################################################################
    # 2. Time and Cell Number Parameters
    ###############################################################################
    
    # For snapshots:
    time_snapshots = [1, 5, 10, 20, 40]    
    
    # Time parameters
    Time_init = 0
    Time_stop = max(time_snapshots)
    steps = int(Time_stop/0.5)+1
    
    # Number of simulations to run
    n_time_snaps = len(time_snapshots)
    N_cells_killed_per_snapshot = 1000
    num_cells = n_time_snaps*N_cells_killed_per_snapshot
    
    # num_cells = 20
    
    ###############################################################################
    # 3. Run Gillespie Simulation
    ###############################################################################
    print('Creating Model')
    
                          
    class Molecules10(Model):
        def __init__(self, parameter_values=None):
            # First call the gillespy2.Model initializer.
            Model.__init__(self, name='Larger network')
            
            
            # Define parameters and reactions for the rates of regulation.
            for regulation in parametersAndICs['modelParameters_act']:
                #print(regulation)
                param = parametersAndICs['modelParameters_act'][regulation]
                #print(param)
                self.add_parameter(Parameter(name=regulation, expression=param))
            
            # Define parameters and reactions for the rates of basal production.
            for basal in parametersAndICs['modelParameters_bas']:
                #print(regulation)
                param = parametersAndICs['modelParameters_bas'][basal]
                #print(param)
                self.add_parameter(Parameter(name=basal, expression=param))
                
    
            # Define parameters and reactions for the rates of degradation
            for degradation in parametersAndICs['modelParameters_deg']:
                #print(degradation)
                param = parametersAndICs['modelParameters_deg'][degradation]
                #print(param)
                self.add_parameter(Parameter(name=degradation, expression=param))
                
            # Define variables for the molecular species representing genes:
            for gene_IC in parametersAndICs['ICs']:
                #print(gene_IC)
                init_cond = parametersAndICs['ICs'][gene_IC]
                #print(init_cond)
                self.add_species(Species(name=gene_IC, initial_value=init_cond))
                
                
            # Define parameters and reactions for the rates of regulation.
            for regulation in parametersAndICs['modelParameters_act']:
                                   
                gene, target = regulation.split('_')
                gene = 'x'+str(gene[1:])
                target = 'x'+str(target)
                #print(gene)
                #print(target)
                self.add_reaction(Reaction(name="r_reg_"+regulation, reactants={}, products={target:1}, propensity_function=regulation+"*"+gene))
            
            # Define parameters and reactions for the rates of basal production.
            for basal in parametersAndICs['modelParameters_bas']:
                gene = 'x'+str(basal[1:])
    
                self.add_reaction(Reaction(name="r_bas_"+gene, reactants={}, products={gene:1}, propensity_function=basal))
    
            # Define parameters and reactions for the rates of degradation
            for degradation in parametersAndICs['modelParameters_deg']:
                gene = 'x'+str(degradation[1:])
                #print(degradation)
                #print(gene)
                self.add_reaction(Reaction(name="r_deg_"+gene, reactants={gene:1}, products={}, propensity_function=degradation+"*"+gene))
    
    
            # Set the timespan for the simulation.
            self.timespan(np.linspace(Time_init, Time_stop, steps))
           
            
    
    
    # 2.2 Run and Plot
    print('Adding data')
    modelGillespie10 = Molecules10(parametersAndICs)
    
    print('Doing a pre-run of the model, the plot will show one trayectory')
    
    
    results_Gillespie = modelGillespie10.run(number_of_trajectories=1, seed = 10)
    # results_Gillespie.plot()
    
    # results_Gillespie.plot(index=0, xaxis_label='Time', xscale='linear', yscale='linear', 
    #                         yaxis_label='mRNA molecules', style='default', title='Gillespie Simulation - One Trajectory', show_title=True, 
    #                         show_legend=True, multiple_graphs=False, included_species_list=[], 
    #                         save_png=figures_directory+'OneTrajectory_allGenes', figsize=(18, 10))
    
    
    
    print('Running Model - This could take a while...')
    
    
    results_Gillespie = modelGillespie10.run(number_of_trajectories=num_cells, seed = 10)
    # results_Gillespie.plot()
    print('Done! Generating plots')
    print('The plots will be saved in: '+figures_directory)
    
    results_Gillespie.plot(index=0, xaxis_label='Time', xscale='linear', yscale='linear', 
                            yaxis_label='mRNA molecules', style='default', title='Gillespie Simulation - One Trajectory', show_title=True, 
                            show_legend=True, multiple_graphs=False, included_species_list=[], 
                            save_png=figures_directory+'OneTrajectory_allGenes', figsize=(18, 10))
    
    
    results_Gillespie.plot_mean_stdev(xscale='linear', yscale='linear', xaxis_label='Time', 
                                      yaxis_label='mRNA molecules', title='Gillespie Simulation - Mean and Std Dev', show_title=True, 
                                      style='default', show_legend=True, included_species_list=[], ddof=0, 
                                      save_png=figures_directory+network_name+'_OneTrajectory_allGenes_mean_std', figsize=(18, 10))
    
    average_results = results_Gillespie.average_ensemble()
    average_results.plot(title="Mean of trajectories", save_png=figures_directory+'MeanTrajectories')
    
    
    
    
    # for gene in genes:
    #     results_Gillespie.plot(index=None, xaxis_label='Time', xscale='linear', yscale='linear', 
    #                             yaxis_label='mRNA molecules', style='default', title='Gillespie Simulation '+gene, show_title=True, 
    #                             show_legend=False, multiple_graphs=False, included_species_list=[gene], 
    #                             save_png=figures_directory+'AllTrajectories_'+gene, figsize=(18, 10))
        
    #     results_Gillespie.plot_mean_stdev(xscale='linear', yscale='linear', xaxis_label='Time', 
    #                                       yaxis_label='Value', title='Gillespie Simulation Mean and Std Dev '+gene, show_title=True, style='default',
    #                                       show_legend=True, included_species_list=[gene], ddof=0, 
    #                                       save_png=figures_directory+'AllTrajectories_mean_std_'+gene, figsize=(18, 10))
    
    
    
    
    
    
    ###############################################################################
    # 4. Save simulation snapshots in digital expression matrix
    ###############################################################################
    
    print('Saving Snapshots:')
    print(time_snapshots)
    print('The snapshots will be saved in: '+result_directory)
    print('As: ')
    
    for snapshot in range(len(time_snapshots)):
        time_snapshot = time_snapshots[snapshot]
    
        
        Gillespie_time = results_Gillespie.data[N_cells_killed_per_snapshot*snapshot:N_cells_killed_per_snapshot*(snapshot+1)]
    
    
        idx_time = np.where(Gillespie_time[0].data['time']==time_snapshot)[0][0]
        d_time = {}
        for gene in genes:
            Gillespie_time_gene = [Gillespie_time[i][gene][idx_time] for i in range(N_cells_killed_per_snapshot)]
            d_time[gene] = Gillespie_time_gene
    
        
        excel_name = '/t'+str(time_snapshot)+'.xlsx'
        txt_name = '/t'+str(time_snapshot)+'.txt'
        
        print(excel_name)
        print(txt_name)
        
        df_G_time = pd.DataFrame(d_time)
        df_G_time.to_excel(result_directory+excel_name, index=False)
    
        df_G_time_t = df_G_time.transpose()
        df_G_time_t = df_G_time_t.add_prefix('c_')
        df_G_time_t.to_csv(result_directory+txt_name, sep=' ')
        
    
    ###############################################################################
    # 5. Save simulation results in pickle file
    ###############################################################################
    
    print('Saving the whole trayectories for all the cells in: ')
    pickle_name = '/Trajectories.p'
    print(pickle_name)
    
    file_save = open(result_directory+pickle_name, 'wb')
    pickle.dump(results_Gillespie, file_save)
    file_save.close()
    
    
    
    ###############################################################################
    # 6. Save true links to evaluate later
    ###############################################################################
    
    print('Saving the true links of the network in: ')
    links_name = '/Links.xlsx'
    print(links_name)
    
    
    links_df = pd.DataFrame(data = links, columns= ['gene1', 'gene2', 'Link'])
    links_df.to_excel(network_directory+links_name)
    
    
    ###############################################################################
    # 7. Save differential equations to evaluate later
    ###############################################################################
    
    print('Saving the differential equations of the network in: ')
    de_name = '/'+network_name+'_differential_equations.txt'
    print(de_name)
    
    file = open(network_directory+de_name, 'w')
    for key, value in differential_equations.items():
        file.write('d%s : %s\n' % (key, value))
    
    file.close()
    
    
    










