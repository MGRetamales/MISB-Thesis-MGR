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
import scipy.stats as stats


working_dir = 'C:/Users/Carlos/Desktop/Thesis'
os.chdir(working_dir)
network_dir = '/Results/BloodData/'

treatments = ['COM', 'ERY', 'MYL', 'ALL']


plt.rc('font', size=12)          # controls default text sizes
plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.rc('legend', fontsize=14)    # legend fontsize
plt.rc('figure', titlesize=20)  # fontsize of the figure title


# Plot treatments separately
# Use all data
# Separated by timepoints
for treatment in treatments:
    
    matrices_directory = working_dir+network_dir+treatment+'/0_Data'
    
    results_directory = working_dir+network_dir+treatment+'/1_scAna_Data'
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
    ncols = 4
    nrows = min(len(genes)//ncols+1*len(genes)%ncols, 5)
    
    all_matrix = pd.DataFrame(columns=genes)
    legend = []
    
    fig1, ax1 = plt.subplots(nrows = nrows, ncols = ncols, figsize=[10*ncols,5*nrows])
    max_expression = 10
    for time in times:
        print(time)
        
        i = times.index(time)
        filename = matrices_directory+'/t'+time+'.txt'
        
        if treatment == 'ALL' and time == '0':
            filename = working_dir+network_dir+'EML/0_Data/t0.txt'
            
        matrix = pd.read_csv(filename, sep = ' ', index_col=0)
        
        legend_t = np.repeat(i,len(matrix.columns))  
        
        all_matrix_it = matrix.transpose()
        
        all_matrix = pd.concat([all_matrix, all_matrix_it], ignore_index = True)  
        legend = np.concatenate((legend,legend_t))              
        
        max_expression = max(matrix.max().max()+10, max_expression) 
        expression[time]=matrix
        
        n_cells = len(matrix.columns)
        row = 0
        col = 0
        for gene in genes: 
            # plt.hist(matrix.loc[gene,:],bins=np.arange(0,max_expression,1), histtype='stepfilled', color=colors[i], alpha=0.8, align='left', density = True)
            # plt.title('Gene '+gene+' at time ' +time+' for '+treatment+ ' treatment')
            # plt.xlabel('mRNA molecules')
            # plt.ylabel('N° of cells')
            # if gene =='Gapdh' and time == '1':
            #     mu = matrix.loc[gene,:].mean()
            #     std = matrix.loc[gene,:].std()
            #     x = np.linspace(mu - 3*std, mu + 3*std, 100)
            #     plt.plot(x, stats.norm.pdf(x, mu, std))
            # plt.savefig(results_directory+'/Genes/'+gene+'_'+time+'_'+treatment+'.png')
            # plt.show()
            if nrows == 1:
                ax1[col].hist(matrix.loc[gene,:],bins=np.arange(0,max_expression,1), histtype='stepfilled', color=colors[i], alpha=0.8, align='left', density = True)
            
            else:
                ax1[row, col].hist(matrix.loc[gene,:],bins=np.arange(0,max_expression,1), histtype='stepfilled', color=colors[i], alpha=0.8, align='left', density = True)
            
            row += 1
            
            if row == nrows:
                col += 1
                row = 0
            
    
    row = 0
    col = 0
    for gene in genes: 
        if nrows == 1:
            ax1[col].set(title=gene, xlabel='mRNA molecules', ylabel='N° of cells')
            ax1[col].legend(times, title='Time', loc="upper right")
        else:
                
            ax1[row, col].set(title=gene, xlabel='mRNA molecules', ylabel='N° of cells')
            ax1[row, col].legend(times, title='Time', loc="upper right")
        row += 1
        if row == nrows:
            col += 1
            row = 0

    
    fig1.tight_layout()
    fig1.delaxes(ax1[-1,-1])
    fig1.savefig(results_directory+'/Data_simulated_times_'+'_'.join(times)+'_'+treatment+'.png')
    plt.show()
    fig1.clear()  
                       
    
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
    plt.legend(shadow=False, scatterpoints=1, title='Time', bbox_to_anchor =(1.2, 1))
    plt.title("PCA of Timepoints for "+treatment+" cells")
    plt.xlabel("PC1 "+pca_1)
    plt.ylabel("PC2 "+pca_2)
    
    plt.savefig(results_directory+'/PCA_'+treatment+'.png' , bbox_inches='tight')


# Separated by cell types
treatments = ['EML', 'COM', 'ERY', 'MYL']
    
matrices_directory = working_dir+network_dir+'treatments/0_Data'

results_directory = working_dir+network_dir+'treatments/1_scAna_Data'
if not os.path.isdir(results_directory):
    os.mkdir(results_directory)

print('The plots will be saved in: '+results_directory)
    
#############################################################################
## DATA - Preload matrices of scRNA seq
#############################################################################

# # Time of snapshots:
files =  os.listdir(matrices_directory)

# Genes in the first file
file_d0_name = matrices_directory+'/EML.txt'
matrix = pd.read_csv(file_d0_name, sep = ' ', index_col=0)
genes = list(matrix.index)
    


expression = {}
colors = cm.get_cmap('tab10', 8).colors
ncols = 4
nrows = min(len(genes)//ncols+1*len(genes)%ncols, 5)

all_matrix = pd.DataFrame(columns=genes)
legend = []

fig1, ax1 = plt.subplots(nrows = nrows, ncols = ncols, figsize=[10*ncols,5*nrows])
max_expression = 10
for treatment in treatments:
    print(treatment)
    
    i = treatments.index(treatment)
    filename = matrices_directory+'/'+treatment+'.txt'
    matrix = pd.read_csv(filename, sep = ' ', index_col=0)
    
    legend_t = np.repeat(i,len(matrix.columns))  
    
    all_matrix_it = matrix.transpose()
    
    all_matrix = pd.concat([all_matrix, all_matrix_it], ignore_index = True)  
    legend = np.concatenate((legend,legend_t))              
    
    max_expression = max(matrix.max().max()+10, max_expression) 
    expression[treatment]=matrix
    
    row = 0
    col = 0
    for gene in genes: 
        if nrows == 1:
            ax1[col].hist(matrix.loc[gene,:],bins=np.arange(0,max_expression,1), histtype='stepfilled', color=colors[i], alpha=0.8, align='left')
        
        else:
            ax1[row, col].hist(matrix.loc[gene,:],bins=np.arange(0,max_expression,1), histtype='stepfilled', color=colors[i], alpha=0.8, align='left')
        
        row += 1
        
        if row == nrows:
            col += 1
            row = 0
        

row = 0
col = 0
for gene in genes: 
    if nrows == 1:
        ax1[col].set(title=gene, xlabel='mRNA molecules', ylabel='N° of cells')
        ax1[col].legend(times, title='Time', loc="upper right")
    else:
            
        ax1[row, col].set(title=gene, xlabel='mRNA molecules', ylabel='N° of cells')
        ax1[row, col].legend(treatments, title='Time', loc="upper right")
    row += 1
    if row == nrows:
        col += 1
        row = 0

fig1.delaxes(ax1[-1,-1])
fig1.tight_layout()

plt.savefig(results_directory+'/Data_simulated_times_'+'_'.join(treatments)+'_.png')
plt.show()
fig1.clear()  
                   

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

for color, i, target_name in zip(colors, [0, 1, 2, 3, 4, 5, 6, 7], treatments):
    plt.scatter(
        X_r[legend == i, 0], X_r[legend == i, 1], color=color, alpha=0.8, lw=lw, label=target_name
    )
plt.legend(shadow=False, scatterpoints=1, title='Treatments', bbox_to_anchor =(1, 1))
plt.title("PCA of Timepoints for all treatments")
plt.xlabel("PC1 "+pca_1)
plt.ylabel("PC2 "+pca_2)

plt.savefig(results_directory+'/PCA_'+'_'.join(treatments)+'_.png', bbox_inches='tight')
















