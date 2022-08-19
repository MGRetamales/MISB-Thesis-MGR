# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 12:13:06 2022

@author: MGR

Wassestein distance
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import special


def prob_dens_fun_x(x, mu, sigma):
    #points = array of n of molecules
    p_i = 1/(np.sqrt(2*np.pi)*sigma) * np.exp(-0.5*((x-mu)/sigma)**2) 
    p_x = sum(p_i)/ mu.size
    
    return p_x


def cummulat_distrib_fun_x(x, mu, sigma):
    #points = array of n of molecules
    P_i =1/2 * special.erfc((mu-x)/np.sqrt(2)/sigma)
    P_x = sum(P_i)/ mu.size
    return P_x
    

    
def wasserstein_distance_1D(points_a, points_b, sigma, Ntrunc, step):
    #points = array of n of molecules
    x = np.arange(0.0, Ntrunc, step)
    P_x_a = np.zeros_like(x).astype(np.float32)
    P_x_b = np.zeros_like(x).astype(np.float32)
    
    for i in range(x.size):
        P_x_a[i] = cummulat_distrib_fun_x(x[i], points_a, sigma)
        P_x_b[i] = cummulat_distrib_fun_x(x[i], points_b, sigma)
    
    wasserstein_distance =  sum(abs(P_x_a -  P_x_b)*step)
    return wasserstein_distance



def wasserstein_distance_plots(points_a, points_b, sigma, names, Ntrunc, step):
    #points = array of n of molecules
    x = np.arange(0.0, Ntrunc, step)
    
    p_x_a = np.zeros_like(x).astype(np.float32)
    p_x_b = np.zeros_like(x).astype(np.float32)
    P_x_a = np.zeros_like(x).astype(np.float32)
    P_x_b = np.zeros_like(x).astype(np.float32)
    
    for i in range(x.size):
        p_x_a[i] = prob_dens_fun_x(x[i], points_a, sigma)
        p_x_b[i] = prob_dens_fun_x(x[i], points_b, sigma)
    
        P_x_a[i] = cummulat_distrib_fun_x(x[i], points_a, sigma)
        P_x_b[i] = cummulat_distrib_fun_x(x[i], points_b, sigma)
    
    plt.plot(x, p_x_a, x, p_x_b)
    plt.title('Probability density function')
    plt.xlabel('x')
    plt.ylabel('p(x)')
    plt.legend(names)
    plt.show()
    
    plt.plot(x, P_x_a, x, P_x_b)
    plt.title('Cumulative density function')
    plt.xlabel('x')
    plt.ylabel('P(x)')
    plt.legend(names)
    plt.show()
    
if __name__ == '__main__':
    
    points_A = np.array([0, 1, 2, 2, 3, 4, 5, 5, 8, 9])
    points_B = np.array([1, 2, 4,5, 8, 20])
    sigma = 1
    names = ['P', 'Q']
    
    Ntrunc = 40
    step = 0.1
    
    
    wasserstein_distance_plots(points_A, points_B, sigma, names, Ntrunc, step)
