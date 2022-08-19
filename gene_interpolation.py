# -*- coding: utf-8 -*-
"""
Created on Tue May 17 19:27:18 2022

@author: MGR

Interpolation function for x1, when the actual matrix is delivered
"""

import numpy as np
from scipy.interpolate import interp1d
import random
from matplotlib import pyplot as plt
from matplotlib import cm

plt.rc('font', size=12)          # controls default text sizes
plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.rc('legend', fontsize=14)    # legend fontsize
plt.rc('figure', titlesize=20)  # fontsize of the figure title

def interpolate_x1(data, index_interpolate, type_interpolation):

    times = []
    x = []
    for timepoint in data:
        t = timepoint
        
        x_val = data[timepoint]
        x_val_int = x_val[:,index_interpolate]
        x_val_s = np.sort(x_val_int)
        
        times.append(t)
        x.append(x_val_s)
        
    # Cubic - spline
    x_array = np.array(x)
    times = list(map(int, times))
    f_c = interp1d(times, x_array, axis = 0, kind = type_interpolation) 
    
    t_start = times[0]
    t_stop = times[-1]
    steps = times[-1]
    dt = float(t_stop - t_start) / steps
    ts = np.arange(t_start, t_stop + dt, dt)

    x_interpolated = f_c(ts)
    
    return x_interpolated



def plot_interpolation(data, index_interpolate, n_cells):
    
    times = []
    x = []
    for timepoint in data:
        t = timepoint
        
        x_val = data[timepoint]
        x_val_int = x_val[:,index_interpolate]
        x_val_s = np.sort(x_val_int)
        
        times.append(t)
        x.append(x_val_s)
    

    # Cubic - spline
    x_array = np.array(x)
    n_cells_max = np.shape(x_array)[1]
    
    cells_sample = random.sample(range(0, n_cells_max), n_cells)
    
    x_array_sample = x_array[:, cells_sample]
    
    times = list(map(int, times))
    f_c = interp1d(times, x_array_sample, axis = 0, kind = 'cubic')
    f_l = interp1d(times, x_array_sample, axis = 0, kind = 'linear')
    f_q = interp1d(times, x_array_sample, axis = 0, kind = 'quadratic')
    
    t_start = times[0]
    t_stop = times[-1]
    steps = times[-1]
    dt = float(t_stop - t_start) / steps
    ts = np.arange(t_start, t_stop + dt, dt)

    x_interpolated_c = f_c(ts)
    x_interpolated_l = f_l(ts)
    x_interpolated_q = f_q(ts)
    
    
    plt.plot(times[0], x_array_sample[0,0], 'ko', label = 'Data')
    plt.plot(times[0], x_array_sample[0, 0], 'k-', label = 'Linear')
    plt.plot(times[0], x_array_sample[0, 0], 'k--', label = 'Quadratic')
    plt.plot(times[0], x_array_sample[0, 0], 'k:', label = 'Cubic')
    
    for cell in range(np.shape(x_array_sample)[1]):
        plt.plot(times, x_array_sample[:,cell], 'o', ts, x_interpolated_l[:,cell], '-', ts, x_interpolated_q[:,cell], '--', ts, x_interpolated_c[:,cell], ':')
        

    plt.legend(bbox_to_anchor=(1.0, 1.0)) 
    plt.xlabel('Time')
    plt.ylabel('Time')
    plt.title('Interpolation for X1')
        




def interpolate_x1_df(data, gene_interpolate, type_interpolation, dt):

    times = []
    x = []
    for timepoint in data:
        
        x_val = data[timepoint]
        
        x_val_int = x_val.loc[gene_interpolate,:]
        x_val_s = np.sort(x_val_int)
        
        times.append(timepoint)
        x.append(x_val_s)
        
    # Cubic - spline
    x_array = np.array(x)
    times = list(map(int, times))
    f_c = interp1d(times, x_array, axis = 0, kind = type_interpolation) 
    
    t_start = times[0]
    t_stop = times[-1]
    
    ts = np.arange(t_start, t_stop + dt, dt)

    x_interpolated = f_c(ts)
    
    return x_interpolated


def plot_interpolation_df(data, gene_interpolate, n_cells, dt):
    
    times = []
    x = []
    for timepoint in data:

        x_val = data[timepoint]
        
        x_val_int = x_val.loc[gene_interpolate,:]
        x_val_s = np.sort(x_val_int)
        
        times.append(timepoint)
        x.append(x_val_s)
    

    # Cubic - spline
    x_array = np.array(x)
    n_cells_max = np.shape(x_array)[1]
    
    cells_sample = random.sample(range(0, n_cells_max), n_cells)
    
    x_array_sample = x_array[:, cells_sample]
    
    times = list(map(int, times))
    f_c = interp1d(times, x_array_sample, axis = 0, kind = 'cubic')
    f_l = interp1d(times, x_array_sample, axis = 0, kind = 'linear')
    f_q = interp1d(times, x_array_sample, axis = 0, kind = 'quadratic')
    
    t_start = times[0]
    t_stop = times[-1]

    ts = np.arange(t_start, t_stop + dt, dt)

    x_interpolated_c = f_c(ts)
    x_interpolated_l = f_l(ts)
    x_interpolated_q = f_q(ts)
    

    colors = cm.get_cmap('tab10', n_cells).colors
    
    fig, ax = plt.subplots(1,  2, figsize=[20,5])

    ax[0].plot(times[0], x_array_sample[0,0], 'ko', label = 'Data')

    for cell in range(np.shape(x_array_sample)[1]):
        ax[0].plot(times, x_array_sample[:,cell], 'o', color=colors[cell])
        

    ax[0].legend(bbox_to_anchor=(1.0, 1.0)) 
    ax[0].set(xlabel='Time',ylabel='N° of mRNA molecules',title='Sorted cells for interpolation')

    
    

    # plt.plot(times[0], x_array_sample[0,0], 'ko', label = 'Data')
    # plt.plot(times[0], x_array_sample[0, 0], 'k-', label = 'Linear')
    # plt.plot(times[0], x_array_sample[0, 0], 'k--', label = 'Quadratic')
    # plt.plot(times[0], x_array_sample[0, 0], 'k:', label = 'Cubic')
    

    # for cell in range(np.shape(x_array_sample)[1]):
    #     plt.plot(times, x_array_sample[:,cell], 'o', color=colors[cell])
    #     plt.plot(ts, x_interpolated_l[:,cell], '-', color=colors[cell])
    #     plt.plot(ts, x_interpolated_q[:,cell], '--', color=colors[cell])
    #     plt.plot(ts, x_interpolated_c[:,cell], ':', color=colors[cell])

           
    # plt.legend(bbox_to_anchor=(1.0, 1.0)) 
    # plt.xlabel('Time')
    # plt.ylabel('N° of mRNA molecules')
    # plt.title('All interpolations for X1')
    # plt.show()
    

    ax[1].plot(times[0], x_array_sample[0,0], 'ko', label = 'Data')
    ax[1].plot(times[0], x_array_sample[0, 0], 'k:', label = 'Cubic')
    

    for cell in range(np.shape(x_array_sample)[1]):
        ax[1].plot(times, x_array_sample[:,cell], 'o', color=colors[cell])
        ax[1].plot(ts, x_interpolated_c[:,cell], ':', color=colors[cell])
        

    ax[1].legend(bbox_to_anchor=(1.0, 1.0)) 
    ax[1].set(xlabel='Time',ylabel='N° of mRNA molecules',title='Cubic interpolation for X1')
    plt.show()


if __name__ == '__main__':
    import pandas as pd
    
    d1 = pd.DataFrame({"x1": np.random.randint(0, 10, 5), "x2": np.random.randint(0, 10, 5)}).T
    d2 = pd.DataFrame({"x1": np.random.randint(5, 15, 5), "x2": np.random.randint(0, 10, 5)}).T
    d3 = pd.DataFrame({"x1": np.random.randint(5, 20, 5), "x2": np.random.randint(5, 20, 5)}).T
    d4 = pd.DataFrame({"x1": np.random.randint(0, 10, 5), "x2": np.random.randint(0, 10, 5)}).T
    
    
    expression = {'0': d1, '5':d2, '10':d3, '20': d4}
    plot_interpolation_df(expression, 'x1', 5, 1)
    
    
    
# x_interpolated = interpolate_x1(expression, 0, 'cubic')
# plot_interpolation(expression, 0, 5)




# x_interpolated_df = interpolate_x1_df(expression, 'x1', 'cubic', 1)
# plot_interpolation_df(expression, 'x1', 5, 1)















