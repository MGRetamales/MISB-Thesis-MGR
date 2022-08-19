# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 08:22:07 2022

@author: MGR

Blood Data - preliminary ana - transform excel in txts
"""
import os 
import numpy as np
import pandas as pd
import pickle


# Directory
working_dir = 'C:/Users/Carlos/Desktop/Thesis'
os.chdir(working_dir)


# Read file
data_file = working_dir+'/BloodData/allcells_sorted.xlsx'
data_all = pd.read_excel(data_file, sheet_name='allcells_sorted')

# Searated Data files 
sep_data_dir = working_dir+'/Results/BloodData'
if not os.path.isdir(sep_data_dir):
    os.mkdir(sep_data_dir)

files_write = {}


### Separate files by samples
samples = data_all.groupby(['Treatment', 'Day']).size().reset_index().rename(columns={0:'count'})

for index, row in samples.iterrows():
    day = row['Day']
    treatment = row['Treatment']
    
    # Get that day and treatment
    data_all_filtered = data_all[data_all.Day==day]
    data_all_filtered = data_all_filtered[data_all_filtered.Treatment==treatment]

    data_all_filtered = data_all_filtered.drop(['Treatment', 'Well','Chip', 'Day', 'Type'], axis = 1)
    data_all_filtered = data_all_filtered.set_index('ID')
    
    # Transpose
    data_all_filtered = data_all_filtered.transpose()
    data_all_filtered = data_all_filtered.add_prefix('c_')
    
    # Save file
    file_dir = sep_data_dir+'/'+treatment
    if not os.path.isdir(file_dir):
        os.mkdir(file_dir)
        
    file_dir = file_dir+'/0_Data'
    if not os.path.isdir(file_dir):
        os.mkdir(file_dir)
        
    file_name = file_dir+'/t'+day[1:]+'.txt'
    data_all_filtered.to_csv(file_name, sep=' ')


## EML is day 0 for all data
d_0_file = sep_data_dir+'/EML/0_Data/t0.txt'
d_0 = pd.read_csv(d_0_file, sep = ' ', index_col=0)

treatments = list(samples.Treatment.unique())
treatments.remove('EML')

for treat in treatments:
    file_dir = sep_data_dir+'/'+treat+'/0_Data'
    file_name = file_dir+'/t0.txt'
    d_0.to_csv(file_name, sep=' ')




## d3 of ERY and MYL has 200 cells instead of 150 
# For now, I will just drop 50 randomly
d_3_ERY_file = sep_data_dir+'/ERY/0_Data/t3.txt'
d_3_ERY = pd.read_csv(d_3_ERY_file, sep = ' ', index_col=0)
d_3_ERY_cols = d_3_ERY.columns
ind_keep = np.random.choice(199, 150, replace=False)
d_3_ERY_cols_new = d_3_ERY_cols[ind_keep]
d_3_ERY_new = d_3_ERY[d_3_ERY_cols_new]
d_3_ERY_new.to_csv(d_3_ERY_file, sep=' ')

d_3_MYL_file = sep_data_dir+'/MYL/0_Data/t3.txt'
d_3_MYL = pd.read_csv(d_3_MYL_file, sep = ' ', index_col=0)
d_3_MYL_cols = d_3_MYL.columns
ind_keep = np.random.choice(199, 150, replace=False)
d_3_MYL_cols_new = d_3_MYL_cols[ind_keep]
d_3_MYL_new = d_3_MYL[d_3_MYL_cols_new]
d_3_MYL_new.to_csv(d_3_MYL_file, sep=' ')






### Save all the treatments and separate them by timepoints only
file_dir = sep_data_dir+'/ALL'
if not os.path.isdir(file_dir):
    os.mkdir(file_dir)

file_dir = file_dir+'/0_Data'
if not os.path.isdir(file_dir):
    os.mkdir(file_dir)

d_1_MYL = pd.read_csv(sep_data_dir+'/MYL/0_Data/t1.txt', sep = ' ', index_col=0)
d_3_MYL = pd.read_csv(sep_data_dir+'/MYL/0_Data/t3.txt', sep = ' ', index_col=0)
d_6_MYL = pd.read_csv(sep_data_dir+'/MYL/0_Data/t6.txt', sep = ' ', index_col=0)

d_1_COM = pd.read_csv(sep_data_dir+'/COM/0_Data/t1.txt', sep = ' ', index_col=0)
d_3_COM = pd.read_csv(sep_data_dir+'/COM/0_Data/t3.txt', sep = ' ', index_col=0)
d_6_COM = pd.read_csv(sep_data_dir+'/COM/0_Data/t6.txt', sep = ' ', index_col=0)

d_1_ERY = pd.read_csv(sep_data_dir+'/ERY/0_Data/t1.txt', sep = ' ', index_col=0)
d_3_ERY = pd.read_csv(sep_data_dir+'/ERY/0_Data/t3.txt', sep = ' ', index_col=0)
d_6_ERY = pd.read_csv(sep_data_dir+'/ERY/0_Data/t6.txt', sep = ' ', index_col=0)


# t0 - repeat EML 3 times
d_0_450 = pd.concat([d_0, d_0, d_0], axis = 1)

d_1_450 = pd.concat([d_1_MYL, d_1_COM, d_1_ERY], axis = 1)
d_3_450 = pd.concat([d_3_MYL, d_3_COM, d_3_ERY], axis = 1)
d_6_450 = pd.concat([d_6_MYL, d_6_COM, d_6_ERY], axis = 1)

    
    
d_0_450.to_csv(sep_data_dir+'/ALL/0_Data/t0.txt', sep=' ')
d_1_450.to_csv(sep_data_dir+'/ALL/0_Data/t1.txt', sep=' ')
d_3_450.to_csv(sep_data_dir+'/ALL/0_Data/t3.txt', sep=' ')
d_6_450.to_csv(sep_data_dir+'/ALL/0_Data/t6.txt', sep=' ')


### Save all the timepoints and separate them by treatments only
file_dir = sep_data_dir+'/treatments'
if not os.path.isdir(file_dir):
    os.mkdir(file_dir)

file_dir = file_dir+'/0_Data'
if not os.path.isdir(file_dir):
    os.mkdir(file_dir)


d_MYL = pd.concat([d_1_MYL, d_3_MYL, d_6_MYL], axis = 1)
d_COM = pd.concat([d_1_COM, d_3_COM, d_6_COM], axis = 1)
d_ERY = pd.concat([d_1_ERY, d_3_ERY, d_6_ERY], axis = 1)


d_0.to_csv(file_dir+'/Control.txt', sep=' ')
d_MYL.to_csv(file_dir+'/MYL.txt', sep=' ')
d_COM.to_csv(file_dir+'/COM.txt', sep=' ')
d_ERY.to_csv(file_dir+'/ERY.txt', sep=' ')















