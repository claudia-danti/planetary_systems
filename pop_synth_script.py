import numpy as np
import multiple_planets_gas_acc as code_gas
import functions_pebble_accretion as peb
from functions import *
import functions_plotting as plot
import matplotlib.pyplot as plt
import matplotlib as mpl
import astropy.units as ub
import pandas as pd
from matplotlib.ticker import ScalarFormatter, LogFormatter, LogLocator, MultipleLocator, AutoMinorLocator
from matplotlib import cm, ticker
from matplotlib import colors
import matplotlib.gridspec as gridspec
import matplotlib.patches as patch
from matplotlib.offsetbox import AnchoredText
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.lines as mlines 
import scipy.stats as stats
import multiprocessing as mp
from scipy.integrate import cumtrapz

color = mpl.colormaps["YlOrRd"].reversed()(np.linspace(0, 0.7, code_gas.sim_params.nr_planets))


# disc parameters
params_dict = {'St_const': None, 
               'iceline_radius': None,
                'alpha': 1e-2,
                'alpha_z': 1e-4, 
                'alpha_frag': 1e-4, 
                'epsilon_el': 1e-2,
                'epsilon_heat':0.5,
                'v_frag': (1 * u.m/u.s).to(u.au/u.Myr).value,
                'M_dot_gas_star': "star_mass_linear",
                'iceline_v_frag_change': True,
                }


# Parameters for the [Fe/H] Gaussian distribution
mu = 0.07  # Mean [Fe/H]
sigma = 0.21  # Standard deviation
num_samples = 1000  # Number of Monte Carlo samples
# Generate random Z values from a Gaussian distribution
Fe_H_samples = np.random.normal(mu, sigma, num_samples)
Z_samples = Fe_H_to_Z(Fe_H_samples)  # Convert Fe/H samples to Z

# random sample initial star masses from the IMF
which = 'Chabrier2005' #'Kroupa'
Mstars = np.logspace(-2, 2, num_samples)
IMF_pdf = np.zeros(num_samples)
MC_random = np.random.uniform(0, 1, num_samples)

for i in range(0, num_samples):
    if which == 'Chabrier2005':
        IMF_pdf[i] = Chabrier_2005_IMF_pdf(Mstars[i])
    if which == 'Kroupa':
        IMF_pdf[i] = Kroupa_IMF_pdf(Mstars[i])    

# Assume x is your array (can be linear or log-spaced), pdf is the unnormalized PDF
dx = np.diff(Mstars)
dx = np.append(dx, dx[-1])  # Make dx same length as x
# Compute normalization constant (area under curve)
area = np.sum(IMF_pdf * dx)
# Normalize
IMF_pdf_norm = IMF_pdf / area
# sample the cdf from the normalized PDF
IMF_cdf = cumtrapz(IMF_pdf_norm, Mstars, initial=0)
IMF_cdf /= IMF_cdf[-1]
mstar_samples = np.interp(MC_random, IMF_cdf, Mstars)


# Gaussian distribution of disc lifetime
mu = 3  # Mean disc lifetime in Myr
sigma = 0.5  # Standard deviation
num_samples = 1000  # Number of Monte Carlo samples
# Generate random tau_disc values from a Gaussian distribution
tau_disc_samples = np.random.normal(mu, sigma, num_samples)

#star_mass = 0.5*const.M_sun.to(u.M_earth).value
output_folder = 'sims/gas_acc/stellar_masses/single_planets/linear/surfheat/lowres/Fe_H_07/5Myrs_randomMstar_randomZ'
t_fin = 5 #Myr, end of sim
N_steps = 500 #number of steps of the sim 

# Number of samples to generate
seed = 12
seed_t0 = 9
see_ap0 = 5

# inner embryo
t0_inner_samples = stats.uniform.rvs(loc=0.1, scale=0.9, size=num_samples, random_state=seed_t0)
R_in = 0.1
R_out = 30
a_p0_inner_samples = stats.loguniform.rvs(R_in, R_out, size=num_samples, random_state=see_ap0)
t0_inner = ([t0_inner_samples] * np.ones(len(a_p0_inner_samples))) # warning, this also goes in the initial conditions when doing mulitple planets otherwise it won't work


# for  a_p0_in, t0_in, Z in zip(a_p0_inner_samples, t0_inner_samples, Z_samples):
#     params = code_gas.Params(**params_dict, H_r_model='Lambrechts_mixed', star_mass=0.1*const.M_sun.to(u.M_earth).value, Z = Z)

for  a_p0_in, t0_in, Z, star_mass in zip(a_p0_inner_samples, t0_inner_samples, Z_samples, mstar_samples ):
    params = code_gas.Params(**params_dict, H_r_model='Lambrechts_mixed', star_mass=star_mass*const.M_sun.to(u.M_earth).value, Z = Z)
    print("Z", params.Z)
    #initial conditions: both a_p0 and m0 take the outer planet and one of the inner planets
    sigma_gas_inner = sigma_gas_steady_state(a_p0_in, t0_in, params)
    m0_in = M0_pla_Mstar(a_p0_in, t0_in, sigma_gas_inner, params)

    a_p0 = np.array([a_p0_in])
    m_0 = np.array([m0_in])
    t0 = np.array([t0_in])

    sim_params_dict = {'N_step': N_steps,
                    'm0': m_0,
                    'a_p0': a_p0,
                    't0': t0,
                    't_fin': t_fin,
                }
    sim_params = code_gas.SimulationParams(**sim_params_dict)
    peb_acc = code_gas.PebbleAccretion(simplified_acc=False)
    gas_acc = peb.GasAccretion()

    result = code_gas.simulate_euler(migration = True, filtering = True, peb_acc = peb_acc, gas_acc=gas_acc, params=params, sim_params=sim_params, output_folder=output_folder)


# # outer embryo
# a_p0_outer = np.array([30])
# t0_outer = ([0.2] * np.ones(len(a_p0_outer)) ) # warning, this also goes in the initial conditions when doing mulitple planets otherwise it won't work
# m0_outer = M0_pla_Mstar(a_p0_outer, t0_outer, sigma_gas_steady_state(a_p0_outer, t0_outer, params), params)

# for  a_p0_in, t0_in in zip(a_p0_inner_samples, t0_inner_samples):

#     #initial conditions: both a_p0 and m0 take the outer planet and one of the inner planets
#     sigma_gas_inner = sigma_gas_steady_state(a_p0_in, t0_in, params)
#     m0_in = M0_pla_Mstar(a_p0_in, t0_in, sigma_gas_inner, params)

#     a_p0 = np.array([*a_p0_outer, a_p0_in])
#     m_0 = np.array([*m0_outer, m0_in])
#     t0 = np.array([*t0_outer, t0_in])

#     sim_params_dict = {'N_step': N_steps,
#                     'm0': m_0,
#                     'a_p0': a_p0,
#                     't0': t0,
#                     't_fin': t_fin,
#                 }
#     sim_params = code_gas.SimulationParams(**sim_params_dict)
#     peb_acc = code_gas.PebbleAccretion(simplified_acc=False)
#     gas_acc = peb.GasAccretion()

#     result = code_gas.simulate_euler(migration = True, filtering = True, peb_acc = peb_acc, gas_acc=gas_acc, params=params, sim_params=sim_params, output_folder=output_folder)
