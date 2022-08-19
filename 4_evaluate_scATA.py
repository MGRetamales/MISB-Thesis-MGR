# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 08:24:08 2022

@author: MGR

Test the results of the OF function with the parameters found
1. Read the data (Again)
2. With the parameters - Get the value of OF (check that it is correct)
3. Plot for each time point/gene the real and estimated
"""


import os 
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm

from modelClass_ODE_SDE_x2 import  dW_matrix
from ATAsc_cost_functions import sde_x2_cf_Scaled_InitNoise_x1int_x2ICs_l2
from gene_interpolation import interpolate_x1_df
# from plot_cost_function import plot_sde_data

working_dir = 'C:/Users/Carlos/Desktop/Thesis'
os.chdir(working_dir)



networks = ['network02_01', 'network02_02', 'network02_05', 'network05_01', 
            'network05_02', 'network05_03', 'network05_04', 'network05_05',
            'network05_06', 'network10_01', 'network10_02', 'network10_03', 'network10_04']

networks = ['COM', 'ERY', 'MYL', 'ALL']



for network_name in networks:
    
    np.random.seed(10)
    #network_name = 'network10_04'
    print("Name of network: "+network_name)
    
    
    network_directory = working_dir+'/Results/00_Whole_Pipelines/'+network_name
    
    matrices_directory = network_directory+'/0_Data'
    ATA_directory = network_directory+'/3_ATAsc'
    
    results_directory = network_directory+'/4_ATAsc_Analysis'
    if not os.path.isdir(results_directory):
        os.mkdir(results_directory)
        
    figures_directory = results_directory+'/Figures'
    if not os.path.isdir(figures_directory):
        os.mkdir(figures_directory)
        
        
    
    plt.rc('font', size=12)          # controls default text sizes
    plt.rc('axes', titlesize=18)     # fontsize of the axes title
    plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
    plt.rc('legend', fontsize=14)    # legend fontsize
    plt.rc('figure', titlesize=20)  # fontsize of the figure title
    
    
    
    #############################################################################
    ## DATA - Preload matrices of scRNA seq
    #############################################################################
    
    
    # # Time of snapshots:
    files = list(filter(lambda x: '.txt' in x, os.listdir(matrices_directory)))
    times = [files[i][1:-4] for i in range(len(files))]
    times.sort(key= int)
    
    # Genes in the first file
    file_0_name = matrices_directory+'/t'+times[0]+'.txt'
    matrix = pd.read_csv(file_0_name, sep = ' ', index_col=0)
    genes = list(matrix.index)
    
    #print('The dataset contains the following genes:')
    #print(genes)
    
    
    #print('All the data will be saved in: '+results_directory)
    #print('Loading data from: '+ matrices_directory)
    
    
    # Save the expression o each time point
    # For now it has to be 1000 cells in each timepoint
    expression = {}
    
    # A long color list for timepoints
    colors = cm.get_cmap('Paired', len(times)).colors
    
    
    # Save and Plot the timepoits to see how they behave... 
    # plot max 10 genes :/
    
    max_expression = 10
    for time in times:
        i = times.index(time)
        filename = matrices_directory+'/t'+time+'.txt'
        matrix = pd.read_csv(filename, sep = ' ', index_col=0)
        
        max_expression = max(matrix.max().max()+10, max_expression) 
        expression[time]=matrix
        
    # Initilize noise matrix - the seed is the same as in 3_ATAsc.py, so the matrix should be the same
    tspan = list(map(int, times))
    timeParameters = {'t_start':tspan[0], 't_stop':tspan[-1], 'steps':tspan[-1]}
    dw_matrix = dW_matrix(3, timeParameters, 1000) 
    
    wd_params = [1, max_expression, 0.1]
    
    
    dt = 1
    
    # Initial time and initial conditions
    initial_timepoint = times[0]
    initial_conditions = expression[initial_timepoint]
    genes = list(initial_conditions.index)
    
    #############################################################################
    #print('Loading regulation results from: '+ ATA_directory)
    files_reg =  os.listdir(ATA_directory)
    files_reg = [x for x in files_reg if x.startswith('REG')]
    
    
    results_ATA = pd.read_excel(ATA_directory+'/'+files_reg[0], index_col = 0)
    
    for file in files_reg[1:]:
        results = pd.read_excel(ATA_directory+'/'+file, index_col = 0)    
        results_ATA = results_ATA.append(results, ignore_index = True)
    
    
    #############################################################################
    #print('Loading without regulation results from: '+ ATA_directory)
    files_noreg =  os.listdir(ATA_directory)
    files_noreg = [x for x in files_noreg if x.startswith('NOREG')]
    
    
    results_ATA_alone = pd.read_excel(ATA_directory+'/'+files_noreg[0], index_col = 0)
    
    for file in files_noreg[1:]:
        results = pd.read_excel(ATA_directory+'/'+file, index_col = 0)    
        results_ATA_alone = results_ATA_alone.append(results, ignore_index = True)
    
    
    
    
    #############################################################################
    # Extract the best values of OF for each combination
    
    best_results = results_ATA.loc[results_ATA.groupby(["gene1","gene2"]).OF.idxmin()]
    best_results_alone = results_ATA_alone.loc[results_ATA_alone.groupby(["gene1","gene2"]).OF.idxmin()]
    
    best_results1 = best_results.drop(columns = ['Solver', 'Scaled', 'Noise', 'Lnorm' ])
    best_results_alone1 = best_results_alone.drop(columns = ['Solver', 'Scaled', 'Noise', 'Lnorm' ])
    
    best_results_compare = best_results1.merge(best_results_alone1, on="gene2", suffixes = ("_reg", "_alone"))
    best_results_compare = best_results_compare.rename(columns={'gene1_reg': 'gene1'})
    
    best_results_compare['a2_unSC'] = 0
    best_results_compare['a3_unSC'] = 0
    best_results_compare['b2_unSC'] = 0
    
    for index, row in best_results_compare.iterrows():
    
        gene1 = row['gene1']
        gene2 = row['gene2']
        
        if gene1 == gene2:
            best_results_compare = best_results_compare.drop(index)
    best_results_compare = best_results_compare.reset_index()
    
    for index, row in best_results_compare.iterrows():
    
        gene1 = row['gene1']
        gene2 = row['gene2']
    
    
        # Do the interpolation for gene1
        gene_interpolated = interpolate_x1_df(expression, gene1, 'cubic', dt)
        gene_x1_interpolated = gene_interpolated*(gene_interpolated >= 0)
    
        # For ICs of the rest of the genes
        # # Sort timepoint1 by gene 1
        data_x0_sorted = initial_conditions.sort_values(by = gene1, axis = 1)
    
    
        print(gene1, gene2) 
        #print(row['OF_reg'])
    
        # Save data of gene2 for the timepoints
        data_gene2 = {}
        for time in times:
            data_gene2_time = expression[time].loc[gene2,:]
            data_gene2[time] = data_gene2_time
           
        # Get the Initial conditions for X1 and X2 for the same cell 
        x1_IC = data_x0_sorted.loc[gene1]
        x2_IC = data_x0_sorted.loc[gene2]
        # n_cells = len(x2_IC)
        
        # Get the mean of the initial condition, to be used for scaling purposes
        mean_IC_x1 = np.mean(x1_IC)
        mean_IC_x2 = np.mean(x2_IC)
        # scaling factor [a2, a3*x1, b2*x3]
        scaling_factors = [1, mean_IC_x1, mean_IC_x2]
    
    
        ab = [row['a2_reg'], row['a3_reg'], row['b2_reg']]
        
        ab_final = np.array(ab)/np.array(scaling_factors)
        #print(ab_final)
        best_results_compare.iat[index,best_results_compare.columns.get_loc('a2_unSC')] = ab_final[0] 
        best_results_compare.iat[index,best_results_compare.columns.get_loc('a3_unSC')] = ab_final[1] 
        best_results_compare.iat[index,best_results_compare.columns.get_loc('b2_unSC')] = ab_final[2] 
        #best_results_compare.loc[index].a3_unSC = ab_final[1] 
        #best_results_compare.loc[index].b2_unSC = ab_final[2]     
       
        OF_new = sde_x2_cf_Scaled_InitNoise_x1int_x2ICs_l2(ab, x2_IC, tspan, data_gene2, wd_params, scaling_factors, dw_matrix, gene_x1_interpolated)
        genes_name = gene2+' regulated by '+gene1
        plots_directory = figures_directory+'/'+gene2+'_regBy'+gene1+'_'
        
        #plot_sde_data(ab, scaling_factors, x2_IC, tspan, data_gene2, dw_matrix, wd_params, gene_x1_interpolated, plots_directory, genes_name)
        
        #if row['OF_reg'] != OF_new:
            #print(row['OF_reg'])
            #print(OF_new)
            #print("THE REG VALUES ARE NOT THE SAME")
    
        #print(gene2)
        #print(row['OF_alone'])
    
        ab_alone = [row['a2_alone'], 0, row['b2_alone']]
        ab_final_alone = np.array(ab_alone)/np.array(scaling_factors)
        
        #print(ab_final_alone)
        
        OF_new_alone = sde_x2_cf_Scaled_InitNoise_x1int_x2ICs_l2(ab_alone, x2_IC, tspan, data_gene2, wd_params, scaling_factors, dw_matrix, gene_x1_interpolated)
        
        if row['OF_alone'] != OF_new_alone:
            #print("THE NOT REG VALUES ARE NOT THE SAME")
            #print(row['OF_alone'])
            #print(OF_new_alone)
            best_results_compare.iat[index,best_results_compare.columns.get_loc('OF_alone')] = OF_new_alone
            #best_results_compare['OF_alone'] = best_results_compare['OF_alone'].replace(row['OF_alone'], OF_new)
            
        
        genes_name = gene2+' not regulated'
        plots_directory = figures_directory+'/'+gene2+'_unreg_'+gene1+'_'
        #plot_sde_data(ab, scaling_factors, x2_IC, tspan, data_gene2, dw_matrix, wd_params, gene_x1_interpolated, plots_directory, genes_name)
        
        
    best_results_compare['Difference'] = best_results_compare['OF_reg']-best_results_compare['OF_alone']
    best_results_compare['Division'] = best_results_compare['OF_reg']/best_results_compare['OF_alone']
    best_results_compare['Improvement'] = best_results_compare['Difference']/best_results_compare['OF_alone']*100
        
    best_results_compare['OF_reg_a3'] = np.where(best_results_compare['a3_reg']!=0,best_results_compare['OF_reg'],100)
    best_results_compare['Improvement_a3'] = np.where(best_results_compare['a3_reg']!=0,best_results_compare['Improvement'],100)
    
    
    
    best_results_compare = best_results_compare.sort_values(['gene2', 'Improvement_a3'])
    
    rank = np.linspace(len(genes)-1, 1, len(genes)-1)
    for j in range(len(genes)-1):
        rank = np.append(rank, np.linspace(len(genes)-1, 1, len(genes)-1))
    
    best_results_compare['Rank'] = rank
    best_results_compare['Improvement_a3_rank'] = best_results_compare['Rank']*best_results_compare['Improvement_a3']
    
    analysis_file  = results_directory+'/ATA_analysis_'+network_name+'.xlsx'
    best_results_compare.to_excel(analysis_file)
            
            
            
    #############################################################################
    # Draw ROC CURVES
    
    # Read from the netwrok the links
    true_network_links = pd.read_excel(network_directory+'/Links.xlsx', index_col = 0)
    best_results_compare_link = best_results_compare.merge(true_network_links, on=["gene1","gene2"])
    best_results_compare_link['Link'] = abs(best_results_compare_link['Link'])
    
    analysis_file  = results_directory+'/ATA_analysis_wLinks_'+network_name+'.xlsx'
    best_results_compare_link.to_excel(analysis_file)
    
    net_positives = sum(best_results_compare_link['Link']==1)
    net_negatives = sum(best_results_compare_link['Link']==0)
    
    
    
    
    manual = False
    if manual:
    
        ##############
        # ROC 0 - Coin (50%)
        ##############
        thr_true_positive_rates_0 = [0, 1]
        thr_false_positive_rates_0 = [0, 1]
    
        ##############
        # ROC 1 - Value OF
        ##############
        thresholds_1 = list(best_results_compare_link['OF_reg'])
        # thresholds_1.sort()
        thresholds_1_v2  = np.linspace(min(thresholds_1)-1, max(thresholds_1)+1, 200)
        
        thr_true_positive_rates_1 = [0.0]
        thr_false_positive_rates_1 = [0.0]
        
        thr_true_positive_rates_1 = []
        thr_false_positive_rates_1 = []
        
        for theshold in thresholds_1_v2:
            print(theshold)
            thr_positives = (best_results_compare_link['OF_reg']<=theshold).array.astype(int)
            thr_negatives = (best_results_compare_link['OF_reg']>theshold).array.astype(int)
            
            thr_true_positives = sum((thr_positives == 1).astype(int) *  (best_results_compare_link['Link']== 1).astype(int))
            thr_false_positives = sum(thr_positives) - thr_true_positives
            
            if net_positives == 0:
                thr_true_positive_rate = 0
            else: 
                thr_true_positive_rate = thr_true_positives / net_positives
            
            if net_negatives == 0:
                thr_false_positive_rate = 0
            else:
                thr_false_positive_rate = thr_false_positives / net_negatives
        
        
            thr_true_positive_rates_1.append(thr_true_positive_rate)
            thr_false_positive_rates_1.append(thr_false_positive_rate)
        
        
        #thr_true_positive_rates_1.append(1.0)
        #thr_false_positive_rates_1.append(1.0)
        
        
        
        
        ##############
        # ROC 2 - Value a3
        ##############
        
        thresholds_2 = list(best_results_compare['a3_reg'])
        thresholds_2.sort(reverse = True)
        thresholds_2  = np.linspace(min(thresholds_2)-1, max(thresholds_2)+1, 200)
        
        thr_true_positive_rates_2 = []
        thr_false_positive_rates_2 = []
        
        for theshold in thresholds_2:
            print(theshold)
            thr_positives = (best_results_compare_link['a3_reg']>=theshold).array.astype(int)
            thr_negatives = (best_results_compare_link['a3_reg']<theshold).array.astype(int)
            
            thr_true_positives = sum((thr_positives == 1).astype(int) *  (best_results_compare_link['Link']== 1).astype(int))
            thr_false_positives = sum(thr_positives) - thr_true_positives
        
            if net_positives == 0:
                thr_true_positive_rate = 0
            else: 
                thr_true_positive_rate = thr_true_positives / net_positives
            
            if net_negatives == 0:
                thr_false_positive_rate = 0
            else:
                thr_false_positive_rate = thr_false_positives / net_negatives
        
        
            thr_true_positive_rates_2.append(thr_true_positive_rate)
            thr_false_positive_rates_2.append(thr_false_positive_rate)
        
        
        #thr_true_positive_rates_2.append(1.0)
        #thr_false_positive_rates_2.append(1.0)
        
        
        ##############
        # ROC 3 - Value OF  - a3!=0
        ##############
        thresholds_3 = list(best_results_compare_link['OF_reg_a3'])
        thresholds_3.sort()
        thresholds_3  = np.linspace(min(thresholds_3)-1, max(thresholds_3)+1, 200)
        
        thr_true_positive_rates_3 = []
        thr_false_positive_rates_3 = []
        
        for theshold in thresholds_3:
            print(theshold)
            thr_positives = (best_results_compare_link['OF_reg_a3']<=theshold).array.astype(int)
            thr_negatives = (best_results_compare_link['OF_reg_a3']>theshold).array.astype(int)
            
            thr_true_positives = sum((thr_positives == 1).astype(int) *  (best_results_compare_link['Link']== 1).astype(int))
            thr_false_positives = sum(thr_positives) - thr_true_positives
            
            if net_positives == 0:
                thr_true_positive_rate = 0
            else: 
                thr_true_positive_rate = thr_true_positives / net_positives
            
            if net_negatives == 0:
                thr_false_positive_rate = 0
            else:
                thr_false_positive_rate = thr_false_positives / net_negatives
        
        
            thr_true_positive_rates_3.append(thr_true_positive_rate)
            thr_false_positive_rates_3.append(thr_false_positive_rate)
        
        
        #thr_true_positive_rates_3.append(1.0)
        #thr_false_positive_rates_3.append(1.0)
        
        
        ##############
        # ROC 4 - Value Difference
        ##############
        
        thresholds_4 = list(best_results_compare['Difference'])
        thresholds_4.sort()
        thresholds_4  = np.linspace(min(thresholds_4)-1, max(thresholds_4)+1, 200)
        
        thr_true_positive_rates_4 = []
        thr_false_positive_rates_4 = []
        
        for theshold in thresholds_4:
            print(theshold)
            thr_positives = (best_results_compare_link['Difference']<=theshold).array.astype(int)
            thr_negatives = (best_results_compare_link['Difference']>theshold).array.astype(int)
            
            thr_true_positives = sum((thr_positives == 1).astype(int) *  (best_results_compare_link['Link']== 1).astype(int))
            thr_false_positives = sum(thr_positives) - thr_true_positives
        
            if net_positives == 0:
                thr_true_positive_rate = 0
            else: 
                thr_true_positive_rate = thr_true_positives / net_positives
            
            if net_negatives == 0:
                thr_false_positive_rate = 0
            else:
                thr_false_positive_rate = thr_false_positives / net_negatives
        
        
            thr_true_positive_rates_4.append(thr_true_positive_rate)
            thr_false_positive_rates_4.append(thr_false_positive_rate)
        
        
        #thr_true_positive_rates_4.append(1.0)
        #thr_false_positive_rates_4.append(1.0)
        
        
        
        ##############
        # ROC 5 - Value Division
        ##############
        
        thresholds_5 = list(best_results_compare['Division'])
        thresholds_5.sort()
        thresholds_5  = np.linspace(min(thresholds_5)-1, max(thresholds_5)+1, 200)
        
        thr_true_positive_rates_5 = []
        thr_false_positive_rates_5 = []
        
        for theshold in thresholds_5:
            print(theshold)
            thr_positives = (best_results_compare_link['Division']<=theshold).array.astype(int)
            thr_negatives = (best_results_compare_link['Division']>theshold).array.astype(int)
            
            thr_true_positives = sum((thr_positives == 1).astype(int) *  (best_results_compare_link['Link']== 1).astype(int))
            thr_false_positives = sum(thr_positives) - thr_true_positives
        
            if net_positives == 0:
                thr_true_positive_rate = 0
            else: 
                thr_true_positive_rate = thr_true_positives / net_positives
            
            if net_negatives == 0:
                thr_false_positive_rate = 0
            else:
                thr_false_positive_rate = thr_false_positives / net_negatives
        
        
            thr_true_positive_rates_5.append(thr_true_positive_rate)
            thr_false_positive_rates_5.append(thr_false_positive_rate)
        
        
        #thr_true_positive_rates_5.append(1.0)
        #thr_false_positive_rates_5.append(1.0)
        
        
        
        
        ##############
        # ROC 6 - Value Improvement   
        ##############
        
        thresholds_6 = list(best_results_compare['Improvement'])
        thresholds_6.sort()
        thresholds_6  = np.linspace(min(thresholds_6)-1, max(thresholds_6)+1, 200)
        
        thr_true_positive_rates_6 = []
        thr_false_positive_rates_6 = []
        
        for theshold in thresholds_6:
            print(theshold)
            thr_positives = (best_results_compare_link['Improvement']<=theshold).array.astype(int)
            thr_negatives = (best_results_compare_link['Improvement']>theshold).array.astype(int)
            
            thr_true_positives = sum((thr_positives == 1).astype(int) *  (best_results_compare_link['Link']== 1).astype(int))
            thr_false_positives = sum(thr_positives) - thr_true_positives
        
            if net_positives == 0:
                thr_true_positive_rate = 0
            else: 
                thr_true_positive_rate = thr_true_positives / net_positives
            
            if net_negatives == 0:
                thr_false_positive_rate = 0
            else:
                thr_false_positive_rate = thr_false_positives / net_negatives
        
        
            thr_true_positive_rates_6.append(thr_true_positive_rate)
            thr_false_positive_rates_6.append(thr_false_positive_rate)
        
        
        #thr_true_positive_rates_6.append(1.0)
        #thr_false_positive_rates_6.append(1.0)
        
        
        ##############
        # Plot all
        
        
        plt.plot(thr_false_positive_rates_0, thr_true_positive_rates_0, 'r')
        plt.plot(thr_false_positive_rates_1, thr_true_positive_rates_1, 'C0')
        plt.plot(thr_false_positive_rates_2, thr_true_positive_rates_2, 'C1')
        plt.plot(thr_false_positive_rates_3, thr_true_positive_rates_3, 'C2')
        plt.plot(thr_false_positive_rates_4, thr_true_positive_rates_4, 'C4')
        plt.plot(thr_false_positive_rates_5, thr_true_positive_rates_5, 'C5')
        plt.plot(thr_false_positive_rates_6, thr_true_positive_rates_6, 'C6')
        
        
        
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC')
        plt.legend(['0.5', 'OF', 'a3', 'OF+a3', 'Dif', 'Div', 'Imp'])
        plt.savefig(results_directory+'/ROC_curve_all.png')
        plt.show()
        
        
        
        
        
        
        plt.plot(thr_false_positive_rates_0, thr_true_positive_rates_0, 'r')
        plt.plot(thr_false_positive_rates_1, thr_true_positive_rates_1, 'C0')
        
        
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC')
        plt.legend(['0.5', 'OF'])
        plt.savefig(results_directory+'/ROC_curve_OF.png')
        plt.show()
        
        
        
        plt.plot(thr_false_positive_rates_0, thr_true_positive_rates_0, 'r')
        plt.plot(thr_false_positive_rates_2, thr_true_positive_rates_2, 'C1')
        
        
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC')
        plt.legend(['0.5', 'a3'])
        plt.savefig(results_directory+'/ROC_curve_a3.png')
        plt.show()
        
        
        
        plt.plot(thr_false_positive_rates_0, thr_true_positive_rates_0, 'r')
        plt.plot(thr_false_positive_rates_3, thr_true_positive_rates_3, 'C2')
        
        
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC')
        plt.legend(['0.5', 'OF+a3'])
        plt.savefig(results_directory+'/ROC_curve_OF_a3.png')
        plt.show()
        
        
        
        plt.plot(thr_false_positive_rates_0, thr_true_positive_rates_0, 'r')
        plt.plot(thr_false_positive_rates_4, thr_true_positive_rates_4, 'C4')
        
        
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC')
        plt.legend(['0.5', 'Dif'])
        plt.savefig(results_directory+'/ROC_curve_Dif.png')
        plt.show()
        
        
        plt.plot(thr_false_positive_rates_0, thr_true_positive_rates_0, 'r')
        plt.plot(thr_false_positive_rates_5, thr_true_positive_rates_5, 'C5')
        
        
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC')
        plt.legend(['0.5', 'Div'])
        plt.savefig(results_directory+'/ROC_curve_Div.png')
        plt.show()
        
        
        
        plt.plot(thr_false_positive_rates_0, thr_true_positive_rates_0, 'r')
        plt.plot(thr_false_positive_rates_6, thr_true_positive_rates_6, 'C6')
        
        
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC')
        plt.legend(['0.5', 'Imp'])
        plt.savefig(results_directory+'/ROC_curve_Imp.png')
        plt.show()
    
    
    ############################################################################ 
    # roc curves with sklearn
    
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
    
    
    
    ##############
    # ROC 0 - Coin (50%)
    ##############
    thr_true_positive_rates_0 = [0, 1]
    thr_false_positive_rates_0 = [0, 1]
    
    
    ##############
    # ROC 1 - Value OF
    ##############
    
    fpr_1, tpr_1, _ = roc_curve(best_results_compare_link['Link'], -best_results_compare_link['OF_reg'])
    roc_auc_1 = roc_auc_score(best_results_compare_link['Link'], -best_results_compare_link['OF_reg'])
    
    
    ##############
    # ROC 2 - Value a3
    ##############
    fpr_2, tpr_2, _ = roc_curve(best_results_compare_link['Link'], abs(best_results_compare_link['a3_reg']))
    roc_auc_2 = roc_auc_score(best_results_compare_link['Link'], abs(best_results_compare_link['a3_reg']))
    
    
    
    ##############
    # ROC 3 - Value OF  - a3!=0
    ##############
    fpr_3, tpr_3, _ = roc_curve(best_results_compare_link['Link'], -best_results_compare_link['OF_reg_a3'])
    roc_auc_3 = roc_auc_score(best_results_compare_link['Link'], -best_results_compare_link['OF_reg_a3'])
    
    
    ##############
    # ROC 4 - Value Difference
    ##############
    
    fpr_4, tpr_4, _ = roc_curve(best_results_compare_link['Link'], -best_results_compare_link['Difference'])
    roc_auc_4 = roc_auc_score(best_results_compare_link['Link'], -best_results_compare_link['Difference'])
    
    
    
    ##############
    # ROC 5 - Value Division
    ##############
    
    fpr_5, tpr_5, _ = roc_curve(best_results_compare_link['Link'], -best_results_compare_link['Division'])
    roc_auc_5 = roc_auc_score(best_results_compare_link['Link'], -best_results_compare_link['Division'])
    
    
    ##############
    # ROC 6 - Value Improvement   
    ##############
    fpr_6, tpr_6, _ = roc_curve(best_results_compare_link['Link'], -best_results_compare_link['Improvement'])
    roc_auc_6 = roc_auc_score(best_results_compare_link['Link'], -best_results_compare_link['Improvement'])
    
    
    ##############
    # ROC 7 - Value Improvement + a3
    ##############
    
    
    fpr_7, tpr_7, _ = roc_curve(best_results_compare_link['Link'], -best_results_compare_link['Improvement_a3'])
    roc_auc_7 = roc_auc_score(best_results_compare_link['Link'], -best_results_compare_link['Improvement_a3'])
    
    
    ##############
    # ROC 8 - Value Imp + Rank
    ##############
    
    
    fpr_8, tpr_8, _ = roc_curve(best_results_compare_link['Link'], -best_results_compare_link['Improvement_a3_rank'])
    roc_auc_8 = roc_auc_score(best_results_compare_link['Link'], -best_results_compare_link['Improvement_a3_rank'])
    
    ##############
    # Plot all
    
    
    plt.plot(thr_false_positive_rates_0, thr_true_positive_rates_0, 'r')
    plt.plot(fpr_1, tpr_1, 'C0')
    plt.plot(fpr_2, tpr_2, 'C1')
    plt.plot(fpr_3, tpr_3, 'C2')
    plt.plot(fpr_4, tpr_4, 'C4')
    plt.plot(fpr_5, tpr_5, 'C5')
    plt.plot(fpr_6, tpr_6, 'C6')
    plt.plot(fpr_7, tpr_7, 'C7')
    plt.plot(fpr_8, tpr_8, 'C8')
    
    
    
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC '+ network_name)
    plt.legend(['0.5', 'OF AUROC='+'{0:.2f}'.format(roc_auc_1), 'a3 AUROC='+'{0:.2f}'.format(roc_auc_2), 
                'OF+a3 AUROC='+'{0:.2f}'.format(roc_auc_3),'Dif AUROC='+'{0:.2f}'.format(roc_auc_4), 
                'Div AUROC='+'{0:.2f}'.format(roc_auc_5), 'Imp AUROC='+'{0:.2f}'.format(roc_auc_6), 
                'Imm+a3 AUROC='+'{0:.2f}'.format(roc_auc_7), 'Imp+a3+rk AUROC='+'{0:.2f}'.format(roc_auc_8)])
    plt.savefig(results_directory+'/'+network_name+'_ROC_curve_all_skl.png')
    plt.show()
    
    
    
    plt.plot(thr_false_positive_rates_0, thr_true_positive_rates_0, 'r')
    plt.plot(fpr_2, tpr_2, 'C1')
    plt.plot(fpr_4, tpr_4, 'C4')
    plt.plot(fpr_5, tpr_5, 'C5')
    plt.plot(fpr_6, tpr_6, 'C6')
    plt.plot(fpr_7, tpr_7, 'C7')
    plt.plot(fpr_8, tpr_8, 'C8')
    
    
    
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC '+ network_name)
    plt.legend(['0.5', 'a3 AUROC='+'{0:.2f}'.format(roc_auc_2), 
                'Dif AUROC='+'{0:.2f}'.format(roc_auc_4), 
                'Div AUROC='+'{0:.2f}'.format(roc_auc_5), 'Imp AUROC='+'{0:.2f}'.format(roc_auc_6), 
                'Imp+a3 AUROC='+'{0:.2f}'.format(roc_auc_7), 'Imp+a3+rk AUROC='+'{0:.2f}'.format(roc_auc_8)])
    plt.savefig(results_directory+'/'+network_name+'_ROC_curve_all_skl_noOF.png')
    plt.show()
    
    
    
    
    plt.plot(thr_false_positive_rates_0, thr_true_positive_rates_0, 'r')
    plt.plot(fpr_6, tpr_6, 'C6')
    plt.plot(fpr_8, tpr_8, 'C8')
    
    
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC '+ network_name)
    plt.legend(['0.5',  
                'Imp AUROC='+'{0:.2f}'.format(roc_auc_6), 
                'Imp+a3+rk AUROC='+'{0:.2f}'.format(roc_auc_8)], loc=4)
    plt.savefig(results_directory+'/'+network_name+'_ROC_curve_imp_impa3rk.png')
    plt.show()
    
    
    
    
    
    plt.plot(thr_false_positive_rates_0, thr_true_positive_rates_0, 'r')
    plt.plot(fpr_1, tpr_1, 'C0')
    
    
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC '+network_name+',  AUROC='+'{0:.3f}'.format(roc_auc_1))
    plt.legend(['0.5', 'OF'])
    plt.savefig(results_directory+'/'+network_name+'_ROC_curve_OF_sk.png')
    plt.show()
    
    
    
    plt.plot(thr_false_positive_rates_0, thr_true_positive_rates_0, 'r')
    plt.plot(fpr_2, tpr_2, 'C1')
    
    
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC '+network_name+',  AUROC='+'{0:.3f}'.format(roc_auc_2))
    plt.legend(['0.5', 'a3'])
    plt.savefig(results_directory+'/'+network_name+'_ROC_curve_a3_sk.png')
    plt.show()
    
    
    
    plt.plot(thr_false_positive_rates_0, thr_true_positive_rates_0, 'r')
    plt.plot(fpr_3, tpr_3, 'C2')
    
    
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC '+network_name+',  AUROC='+'{0:.3f}'.format(roc_auc_3))
    plt.legend(['0.5', 'OF+a3'])
    plt.savefig(results_directory+'/'+network_name+'_ROC_curve_OF_a3_sk.png')
    plt.show()
    
    
    
    plt.plot(thr_false_positive_rates_0, thr_true_positive_rates_0, 'r')
    plt.plot(fpr_4, tpr_4, 'C4')
    
    
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC '+network_name+',  AUROC='+'{0:.3f}'.format(roc_auc_4))
    plt.legend(['0.5', 'Dif'])
    plt.savefig(results_directory+'/'+network_name+'_ROC_curve_Dif_sk.png')
    plt.show()
    
    
    plt.plot(thr_false_positive_rates_0, thr_true_positive_rates_0, 'r')
    plt.plot(fpr_5, tpr_5, 'C5')
    
    
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC '+network_name+',  AUROC='+'{0:.3f}'.format(roc_auc_5))
    plt.legend(['0.5', 'Div'])
    plt.savefig(results_directory+'/'+network_name+'_ROC_curve_Div_sk.png')
    plt.show()
    
    
    
    plt.plot(thr_false_positive_rates_0, thr_true_positive_rates_0, 'r')
    plt.plot(fpr_6, tpr_6, 'C6')
    
    
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC '+network_name+',  AUROC='+'{0:.3f}'.format(roc_auc_6))
    plt.legend(['0.5', 'Imp'])
    plt.savefig(results_directory+'/'+network_name+'_ROC_curve_Imp_sk.png')
    plt.show()
    
    
    plt.plot(thr_false_positive_rates_0, thr_true_positive_rates_0, 'r')
    plt.plot(fpr_7, tpr_7, 'C7')
    
    
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC '+network_name+',  AUROC='+'{0:.3f}'.format(roc_auc_7))
    plt.legend(['0.5', 'Imp+a3'])
    plt.savefig(results_directory+'/'+network_name+'_ROC_curve_Imp_a3_sk.png')
    plt.show()
    
    
    plt.plot(thr_false_positive_rates_0, thr_true_positive_rates_0, 'r')
    plt.plot(fpr_8, tpr_8, 'C8')
    
    
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC '+network_name+',  AUROC='+'{0:.3f}'.format(roc_auc_8))
    plt.legend(['0.5', 'Imp+a3+rk'])
    plt.savefig(results_directory+'/'+network_name+'_ROC_curve_Imp_a3_rk_sk.png')
    plt.show()
    
    
    
    scores = ['OF', 'a3', 'OF+a3', 'Dif', 'Div', 'Imp', 'Imp+a3', 'Imp+a3+rk']
    auroc = [roc_auc_1, roc_auc_2, roc_auc_3, roc_auc_4, roc_auc_5, roc_auc_6, roc_auc_7, roc_auc_8]
    
    scores_df = pd.DataFrame(auroc, index = scores, columns = ['AUROC'])
    
    print(network_name)
    print(scores_df)
    auroc_file  = results_directory+'/ATA_analysis_'+network_name+'_roc.xlsx'
    scores_df.to_excel(auroc_file)
            
    
    
    
    















































