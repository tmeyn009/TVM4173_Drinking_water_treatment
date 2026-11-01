# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 20:20:53 2023
@author: Thomas Meyn

This module contains functions relevant for calculations related to filtration
models, like the YAO-, RT- and TE-model. The following functions are included:
    
calc_gamma: Calculating gamma from epsilon
calc_porosity_function: Calculating As from gamma


"""

# Basic functions calculating needed parameters

def calc_gamma(porosity): 
    """ Calculting and returning the porosity coefficient
    gamma, from the porosity of the filter media """
    gamma = (1 - porosity) ** (1/3)
    
    return gamma


def calc_porosity_function(gamma):
    """ Calculation of the porosity function As from
    porosity coefficient gamma, and returning value"""
    porosity_function = (2 * (1 - gamma** 5)) / (2 - 3 * gamma
    + 3 * gamma ** 5 - 2 * gamma ** 6)
    
    return porosity_function

#Yao transport efficiencies for interception, gravity and diffusion

def interception_Yao(particle_size_range, collector_diam):
    """ Calculation of transport efficiency for interception for YAO model"""
    n_I = 1.5 * ((particle_size_range / collector_diam) ** 2)
    
    return n_I

def gravity_Yao(particle_density, particle_size_range, filtration_rate):
    """ Calculation of transport efficiency for gravity for YAO model"""
    n_G = (G * (particle_density - WATER_DENS) * (
        particle_size_range ** 2)) / (18 * MU * filtration_rate)
    
    return n_G

def diffusion_Yao(particle_size_range, collector_diam, filtration_rate, temp, PI_VALUE): 
    """ Calculation of transport efficiency for interception for YAO model"""
    temp_kelvin = temp + 273.15
    Pe = (3 * np.pi * MU * particle_size_range * collector_diam
          * filtration_rate) / (BOLZ * temp_kelvin)
    n_D = 4 * Pe ** (-2/3)
    
    return n_D

def transport_efficiency_Yao(particle_size_range, collector_diam, filtration_rate, temp, PI_VALUE):
        
    n_D = diffusion_Yao(particle_size_range, collector_diam, filtration_rate, temp)
    #n_G = gravity_Yao(particle_size_range, collector_diam, filtration_rate)
    #n_I = interception_Yao(particle_size_range, collector_diam)
    #n = n_D + n_G + n_I
    #return n, n_D, n_G, n_I
    return n_D

""" #RT-model transport efficiencies for interception, gravity and diffusion
def interception_RT(d_p = particle_size_range, v_F = filtration_rate, d_c = collector): 
    N_a = HAM / (3 * np.pi * MU *  d_p ** 2 * v_F)
    N_r = d_p / d_c
    n_I = A_s * ((4 /3)* N_a) ** (1 / 8) * N_r ** (15 / 8)
    return n_I 

def gravity_RT(d_p = particle_size_range, rho_p = particle_density, v_F = filtration_rate, d_c = collector): 
    N_r = d_p / d_c
    N_g = (G * (rho_p - WATER_DENS) * d_p ** 2) / (18 * MU * v_F)
    n_G = 0.00338 * A_s * N_r ** (-0.4) * N_g ** (1.2)
    return n_G

def diffusion_RT(d_p = particle_size_range, d_c = collector, T = temp, v = 10): 
    t_k = T + 273
    Pe = (3 * np.pi * MU * d_p * d_c * v) / (BOLZ * t_k)
    n_D = 4 * A_s ** (1 / 3) * Pe ** (-2/3)
    return n_D

#TE-model transport efficiencies for interception, gravity and diffusion
def interception_TE(d_p = particle_size_range, v_F = filtration_rate, d_c = collector): 
    N_a = HAM / (3 * np.pi * MU *  d_p ** 2 * v_F)
    N_r = d_p / d_c
    n_I = 0.55 * A_s * N_a ** (1 / 8) * N_r ** (1.675)
    return n_I

def gravity_TE(d_p = particle_size_range, d_c = collector, T = temp, v_F = filtration_rate, rho_p = particle_density):
    t_k = T + 273
    N_r = d_p / d_c
    N_vdw = HAM / (BOLZ * t_k)
    N_g = (G * (rho_p - WATER_DENS) * d_p ** 2) / (18 * MU * v_F)
    n_G = 0.22 * N_r ** (-0.24) * N_vdw ** (0.053) * N_g ** (1.11)
    return n_G
    
def diffusion_TE(d_p = particle_size_range, d_c = collector, T = temp, v_F = filtration_rate):
    t_k = T + 273
    N_r = d_p / d_c
    N_vdw = HAM / (BOLZ * t_k)
    Pe = (3 * np.pi * MU * d_p * d_c * v_F) / (BOLZ * t_k)
    n_D = 2.4 * A_s ** (1 / 3) * N_r ** (-0.081) * N_vdw ** (0.052) * Pe ** (-0.715)
    return n_D

#total transport efficiencies for all models 



def transport_efficiency_RT(): 
    n_I = interception_RT()
    n_G = gravity_RT()
    n_D = diffusion_RT()
    n = n_I + n_G + n_D
    return n, n_D, n_G, n_I

def transport_efficiency_TE(): 
    n_I = interception_TE()
    n_G = gravity_TE()
    n_D = diffusion_TE()
    n = n_I + n_G + n_D
    return n, n_D, n_G, n_I """