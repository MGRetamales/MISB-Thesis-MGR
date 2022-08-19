# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 10:35:46 2022

@author: MGR

sc-analysis for Gillespy data
"""


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.decomposition import PCA

#Plots config sizes:
plt.rc('font', size=12)          # controls default text sizes
plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.rc('legend', fontsize=14)    # legend fontsize
plt.rc('figure', titlesize=20)  # fontsize of the figure title

## Important directory, has to be changed in each computer
working_dir = 'C:/Users/Carlos/Desktop/Thesis'
os.chdir(working_dir)

networks = ['network02_01', 'network02_02', 'network02_03', 'network02_04', 'network02_05', 'network05_01', 
            'network05_02', 'network05_03', 'network05_04', 'network05_05',
            'network05_06', 'network10_01', 'network10_02', 'network10_03', 'network10_04']




for network_name in networks:

    matrices_directory = working_dir+'/Results/00_Whole_Pipelines/'+network_name+'/0_Data'
    
    
    results_directory = working_dir+'/Results/00_Whole_Pipelines/'+network_name+'/1_scAna_Data'
    if not os.path.isdir(results_directory):
        os.mkdir(results_directory)
    
    print('The plots will be saved in: '+results_directory)
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
        
    
    
    expression = {}
    colors = cm.get_cmap('tab10', 8).colors
    ncols = 2
    nrows = min(len(genes)//ncols+1*len(genes)%ncols, 10)
    
    all_matrix = pd.DataFrame(columns=genes)
    legend = []
    
    fig, ax = plt.subplots(nrows = nrows, ncols = ncols, figsize=[10*ncols,5*nrows])
    max_expression = 10
    for time in times:
        print(time)
        
        i = times.index(time)
        filename = matrices_directory+'/t'+time+'.txt'
        matrix = pd.read_csv(filename, sep = ' ', index_col=0)
        
        legend_t = np.repeat(i,1000)  
        
        all_matrix_it = matrix.transpose()
        
        all_matrix = pd.concat([all_matrix, all_matrix_it], ignore_index = True)  
        legend = np.concatenate((legend,legend_t))              
        
        max_expression = max(matrix.max().max()+1, max_expression) 
        expression[time]=matrix
        
        row = 0
        col = 0
        for gene in genes: 
            if nrows == 1:
                ax[col].hist(matrix.loc[gene,:],bins=np.arange(0,max_expression,1), histtype='stepfilled', color=colors[i], alpha=0.8, align='left')
            
            else:
                ax[row, col].hist(matrix.loc[gene,:],bins=np.arange(0,max_expression,1), histtype='stepfilled', color=colors[i], alpha=0.8, align='left')
            
            row += 1
            
            if row == nrows:
                col += 1
                row = 0
            
    
    row = 0
    col = 0
    for gene in genes: 
        if nrows == 1:
            ax[col].set(title=gene, xlabel='mRNA molecules', ylabel='N° of cells')
            ax[col].legend(times, title='Time', loc="upper right")
    
        else:
            ax[row, col].legend(times, title='Time', loc="upper right")  
            ax[row, col].set(title=gene, xlabel='mRNA molecules', ylabel='N° of cells')
        row += 1
        if row == nrows:
            col += 1
            row = 0
    
    # if nrows == 1:
        
    #     ax[0].legend(times, title='Time', loc="upper right")
    # else:
    #     ax[0,0].legend(times, title='Time', loc="upper right")
    
    fig.tight_layout()
    
    if row < nrows and not (nrows == 1 or nrows == 5):
        fig.delaxes(ax[row,col])
    
    plt.savefig(results_directory+'/'+network_name+'_Data_simulated_times_'+'_'.join(times)+'.png')
    plt.show()
    fig.clear()  
                       
    
    all_matrix_arr = all_matrix.to_numpy()
    
    
    ## PLOT PCA
    
    pca = PCA(n_components=2)
    X_r = pca.fit(all_matrix_arr).transform(all_matrix_arr)
    
    # Percentage of variance explained for each components
    print(
        "explained variance ratio (first two components): %s"
        % str(pca.explained_variance_ratio_)
    )
    
    pca_1 = str(pca.explained_variance_ratio_[0]*100)[:5]+'%'
    pca_2 = str(pca.explained_variance_ratio_[1]*100)[:5]+'%'
    
    
    plt.figure()
    
    colors = cm.get_cmap('tab10', 8).colors
    
    lw = 2
    
    for color, i, target_name in zip(colors, [0, 1, 2, 3, 4, 5, 6, 7], times):
        plt.scatter(
            X_r[legend == i, 0], X_r[legend == i, 1], color=color, alpha=0.8, lw=lw, label=target_name
        )
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title("PCA of Timepoints")
    plt.xlabel("PC1 "+pca_1)
    plt.ylabel("PC2 "+pca_2)
    
    plt.savefig(results_directory+'/'+network_name+'_PCA.png')

