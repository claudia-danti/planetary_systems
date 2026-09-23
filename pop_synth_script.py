import numpy as np
import multiple_planets_gas_acc as code_gas
import functions_pebble_accretion as peb
from functions import *
import functions_plotting as plot
import astropy.units as ub
import pandas as pd
import scipy.stats as stats
import multiprocessing as mp
from scipy.integrate import cumulative_trapezoid as cumtrapz 
# disc parameters
params_dict = {'St_const': None, 
               'iceline_radius': None,
                'alpha': 1e-2,
                'alpha_z': 1e-4, 
                'alpha_frag': 1e-4, 
                'epsilon_el': 1e-2,
                'epsilon_heat':0.5,
                'v_frag': (1 * u.m/u.s).to(u.au/u.Myr).value,
                'M_dot_gas_star': "Hartmann_2016",
                'M_dot_star_downscale': 1/2,
                'migrationI_downscale': 1/2,
                'iceline_v_frag_change': True,
                'M_dot_star_scatter': True,
                'kappa':(1*u.m**2/u.kg).to(u.au**2/u.M_earth).value #default is (0.005*u.m**2/u.kg).to(u.au**2/u.M_earth).value
                }

# ------------------------------sims parameters------------------------------
output_folder = 'sims/opacities/1Msun/Mdot_scatter/double_planet/100_pairs/1_2_scales_closer'
N_steps = 5000 #number of steps of the sim 
num_samples = 1000  # Number of Monte Carlo samples = number of simulations to run
# seeds for random sampling reproducibility
seed = 72
seed_t0 = 63
seed_ap0 = 28


# ------------------------------ Metallicity sampling ------------------------------
# Parameters for the [Fe/H] Gaussian distribution
mu = -0.02  # Mean [Fe/H]
sigma = 0.22  # Standard deviation
# Generate random Z values from a Gaussian distribution
Fe_H_samples = np.random.normal(mu, sigma, num_samples)
Z_samples = Fe_H_to_Z(Fe_H_samples)  # Convert Fe/H samples to Z

# ------------------------------ Star mass sampling from the IMF ------------------------------
# # random sample initial star masses from the IMF
# which = 'Chabrier2005' #'Kroupa'
# Mstars = np.geomspace(0.01, 5, num_samples) #restricting the stellar sample
# IMF_pdf = np.zeros(num_samples)
# MC_random = np.random.uniform(0, 1, num_samples)
# for i in range(0, num_samples):
#     if which == 'Chabrier2005':
#         IMF_pdf[i] = Chabrier_2005_IMF_pdf(Mstars[i])
#     if which == 'Kroupa':
#         IMF_pdf[i] = Kroupa_IMF_pdf(Mstars[i])    
# # Assume x is your array (can be linear or log-spaced), pdf is the unnormalized PDF
# dx = np.diff(Mstars)
# dx = np.append(dx, dx[-1])  # Make dx same length as x
# # Compute normalization constant (area under curve)
# area = np.sum(IMF_pdf * dx)
# # Normalize
# IMF_pdf_norm = IMF_pdf / area
# # sample the cdf from the normalized PDF
# IMF_cdf = cumtrapz(IMF_pdf_norm, Mstars, initial=0)
# IMF_cdf /= IMF_cdf[-1]
# mstar_samples = np.interp(MC_random, IMF_cdf, Mstars)

# ------------------------------ Disc lifetime sampling ------------------------------
# Gaussian distribution of disc lifetime
mu = 5  # Mean disc lifetime in Myr
sigma = 2  # Standard deviation
# Generate random tau_disc values from a Gaussian distribution
tau_disc_samples = np.random.normal(mu, sigma, num_samples)

# ---------------------- Gas accretion rate scatter -------------------
scatter_dex = 0.5
rng = np.random.default_rng(42)
# one offset per simulation, drawn ONCE
Mdot_scatter_samples = rng.normal(0.0, scatter_dex, num_samples)

# ---------------------- Initial conditions (position and insertion time) -------------------
# initial positions and times for the planets, sampled from uniform and loguniform distributions
t_0 = stats.uniform.rvs(loc=0.1, scale=0.9, size=num_samples, random_state=seed_t0)
R_in = 0.1
R_out = 30
a_p0_samples = stats.loguniform.rvs(R_in, R_out, size=num_samples, random_state=seed_ap0)
t0_samples = (t_0 * np.ones(len(a_p0_samples))) # warning, this also goes in the initial conditions when doing mulitple planets otherwise it won't work


# -----------------------------------------------------
# SINGLE PLANET SIMULATIONS
# -----------------------------------------------------

# for  a_p0, t0, t_fin, Z, Mdot_scatter  in zip(a_p0_samples, t0_samples, tau_disc_samples, Z_samples, Mdot_scatter_samples):

#     params = code_gas.Params(**params_dict, H_r_model='Lambrechts_mixed', star_mass=1*const.M_sun.to(u.M_earth).value, Z = Z, Mdot_star_scatter = Mdot_scatter)
#     mdot_star = M_dot_star(t0, params)
#     sigma_gas_inner = sigma_gas_steady_state(a_p0, H_R(a_p0, mdot_star, params), mdot_star, params)
#     m0 = M0_pla_Mstar(a_p0, H_R(a_p0, mdot_star, params), sigma_gas_inner, params)

#     # initial conditions for the simulation, must be arrays
#     a_p0 = np.array([a_p0])
#     m_0 = np.array([m0])
#     t_0 = np.array([t0])

#     sim_params_dict = {'N_step': N_steps,
#                     'm0': m_0,
#                     'a_p0': a_p0,
#                     't0': t_0,
#                     't_fin': t_fin,
#                 }
#     sim_params = code_gas.SimulationParams(**sim_params_dict)
#     peb_acc = code_gas.PebbleAccretion(simplified_acc=False)
#     gas_acc = peb.GasAccretion()

#     result = code_gas.simulate_euler(migration = True, filtering = True, peb_acc = peb_acc, gas_acc=gas_acc, params=params, sim_params=sim_params, output_folder=output_folder)

# -----------------------------------------------------
# MULTIPLE PLANET SIMULATIONS
# -----------------------------------------------------
# ---------------------- Initial conditions (position and insertion time) -------------------
# n_planets = 5
# # random sample the positions
# low, high = 0.1, 30  # actual value range
# log_low, log_high = np.log10(low), np.log10(high)

# for  t_fin, Z, Mdot_scatter  in zip(tau_disc_samples, Z_samples, Mdot_scatter_samples):
#     params = code_gas.Params(**params_dict, H_r_model='Lambrechts_mixed', star_mass=1*const.M_sun.to(u.M_earth).value, Z = Z, Mdot_star_scatter = Mdot_scatter)

#     #initial conditions: each N_planets systems drwas from the same disc
#     a_p0_embryos = 10 ** np.random.uniform(log_low, log_high, n_planets) #loguniform random
#     a_p0_embryos = np.sort(a_p0_embryos)[::-1] # VERY IMPORTANT, the planets need to be outermost to innermost

#     t0_in = (stats.uniform.rvs(loc=0.1, scale=0.9, size=n_planets, random_state=seed_t0) * np.ones(len(a_p0_embryos))) # warning, this also goes in the initial conditions when doing mulitple planets otherwise it won't work
#     mdot_star = M_dot_star(t0_in, params)
#     sigma_gas_inner =sigma_gas_steady_state(a_p0_embryos, H_R(a_p0_embryos, mdot_star, params), mdot_star, params)
#     m0_embryos = M0_pla_Mstar(a_p0_embryos, H_R(a_p0_embryos, mdot_star, params), sigma_gas_inner, params)

#     sim_params_dict = {'N_step': N_steps,
#                     'm0': m0_embryos,
#                     'a_p0': a_p0_embryos,
#                     't0': t0_in,
#                     't_fin': t_fin,
#                 }
#     sim_params = code_gas.SimulationParams(**sim_params_dict)
#     peb_acc = code_gas.PebbleAccretion(simplified_acc=False)
#     gas_acc = peb.GasAccretion()

#     result = code_gas.simulate_euler(migration = True, filtering = True, peb_acc = peb_acc, gas_acc=gas_acc, params=params, sim_params=sim_params, output_folder=output_folder)

# -----------------------------------------------------
# TWO PLANET SIMULATIONS
# -----------------------------------------------------
num_samples = 100
# outer embryo random sampled between 10 and 30 au, inner embryo random sampled between 0.1 and 10 au
R_in_outer = 1
R_out_outer = 30
a_p0_outer_sample = stats.loguniform.rvs(R_in_outer, R_out_outer, size=num_samples, random_state=19)
R_in = 0.1
R_out = 1
a_p0_inner_sample = stats.loguniform.rvs(R_in, R_out, size=num_samples, random_state=99)
n_planets = 2
for  a_p0_outer, a_p0_inner, t_fin, Z, Mdot_scatter  in zip(a_p0_outer_sample, a_p0_inner_sample, tau_disc_samples, Z_samples, Mdot_scatter_samples):
    a_p0 = np.array([a_p0_outer, a_p0_inner])
    params = code_gas.Params(**params_dict, H_r_model='Lambrechts_mixed', star_mass=1*const.M_sun.to(u.M_earth).value, Z = Z, Mdot_star_scatter = Mdot_scatter)
    t0 = (stats.uniform.rvs(loc=0.1, scale=0.9, size=1, random_state=seed_t0) * np.ones(len(a_p0)))
    # inner edge of the disc, uses Mdot withouth photoevap else it's an implicit equation to solve for R_in
    # and the photoevaporation does not dominate at the beginning of the disc evolution (so valid approx)
    r_in =r_magnetic_cavity(M_dot_star_t(t0), params) 
    mdot_star = M_dot_star_t_photoevap(t0, r_in,params)
    sigma_gas = sigma_gas_steady_state(a_p0, H_R(a_p0, mdot_star, params), mdot_star, params)
    m0 = M0_pla_Mstar(a_p0, H_R(a_p0, mdot_star, params), sigma_gas, params)

    sim_params_dict = {'N_step': N_steps,
                    'm0': m0,
                    'a_p0': a_p0,
                    't0': t0,
                    't_fin': t_fin,
                }
    sim_params = code_gas.SimulationParams(**sim_params_dict)
    peb_acc = code_gas.PebbleAccretion(simplified_acc=False)
    gas_acc = peb.GasAccretion()

    result = code_gas.simulate_euler(migration = True, filtering = True, peb_acc = peb_acc, gas_acc=gas_acc, params=params, sim_params=sim_params, output_folder=output_folder)
