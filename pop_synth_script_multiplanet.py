import numpy as np
import multiple_planets_gas_acc as code_gas
import functions_pebble_accretion as peb
from functions import *
import matplotlib.pyplot as plt
import matplotlib as mpl
import astropy.units as ub
import scipy.stats as stats
import multiprocessing as mp
from numpy.random import default_rng

color = mpl.colormaps["YlOrRd"].reversed()(np.linspace(0, 0.7, code_gas.sim_params.nr_planets))

# disc parameters
params_dict = {'St_const': None, 
               'iceline_radius': None,
                'alpha': 1e-2,
                'alpha_z': 1e-4, 
                'alpha_frag': 1e-4, 
                'epsilon_el': 1e-2,
                'epsilon_heat': 0.5,
                'v_frag': (1 * u.m/u.s).to(u.au/u.Myr).value,
                'M_dot_gas_star': "star_mass_linear",
                'iceline_v_frag_change': True,
                'M_dot_gas_star': "star_mass_linear",
                'H_r_model':'Lambrechts_mixed',
                }


output_folder = 'sims/new/test2'
N_steps = 5000 #number of steps of the sim 
num_samples = 100
# Number of samples to generate
seed = 35
seed_t0 = 87
seed_ap0 = 119
seed_Z = 93
# Parameters for the [Fe/H] Gaussian distribution
mu = -0.02  # Mean [Fe/H]
sigma = 0.22  # Standard deviation

rng = np.random.default_rng(seed_Z)
# Generate random Z values from a Gaussian distribution
Fe_H_samples = rng.normal(mu, sigma, num_samples)
Z_samples = Fe_H_to_Z(Fe_H_samples)  # Convert Fe/H samples to Z


# Gaussian distribution of disc lifetime
mu = 5  # Mean disc lifetime in Myr
sigma = 0.5  # Standard deviation
num_samples = 1000  # Number of Monte Carlo samples
# Generate random tau_disc values from a Gaussian distribution
tau_disc_samples = np.random.normal(mu, sigma, num_samples)

t_fin = 5 #Myr, end of sim

num_planets = 4
R_in = 0.1
R_out = 30
rng_cond = default_rng(26)
#loop over the number of simulations (each 4 planets draw from the same Z, Mstar etc samples)
for  Z in Z_samples:
    # planet embyos initial conditions
    a_p0_planets = stats.loguniform.rvs(R_in, R_out, size=num_planets, random_state = rng_cond)
    a_p0_planets = np.sort(a_p0_planets)[::-1] #very important, the planets need to be outermost to innermost
    print("ap0", a_p0_planets)
    t0_samples = stats.uniform.rvs(loc=0.1, scale=0.9, size=num_planets, random_state = rng_cond)

    params = code_gas.Params(**params_dict, star_mass=0.2*const.M_sun.to(u.M_earth).value, Z = Z)
    t0_planets = (t0_samples * np.ones(len(a_p0_planets))) # warning, this also goes in the initial conditions when doing mulitple planets otherwise it won't work
    #initial conditions: both a_p0 and m0 take the outer planet and one of the inner planets
    mdotstar = M_dot_star(t0_planets, params)
    Hr = H_R(a_p0_planets, mdotstar, params)
    sigma_gas = sigma_gas_steady_state(a_p0_planets, Hr, mdotstar, params)
    m0_planets = M0_pla_Mstar(a_p0_planets, t0_planets, sigma_gas, params)
    a_p0 = np.array(a_p0_planets)
    m_0 = np.array(m0_planets)
    t0 = np.array(t0_planets)

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
