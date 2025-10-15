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
                'epsilon_el': 1,
                'epsilon_heat':1.,
                'v_frag': (1 * u.m/u.s).to(u.au/u.Myr).value,
                'M_dot_gas_star': "star_mass_linear",
                'Z': 0.01,
                'H_r_model':'irradiated',
                }


output_folder = 'sims/gas_acc/stellar_masses/multiplanet/linear/irradiated/test'
t_fin = 3 #Myr, end of sim
N_steps = 5000 #number of steps of the sim 
num_samples = 50
# Number of samples to generate
seed = 42
seed_t0 = 11
seed_ap0 = 7


num_planets = 4
R_in = 0.1
R_out = 30
params = code_gas.Params(**params_dict)

for  sim in range(num_samples):
    # inner embryo
    a_p0_planets = stats.loguniform.rvs(R_in, R_out, size=num_planets)
    a_p0_planets = np.sort(a_p0_planets)[::-1] #very important, the planets need to be outermost to innermost
    print("ap0", a_p0_planets)
    t0_samples = stats.uniform.rvs(loc=0.1, scale=0.9, size=num_planets)

    t0_planets = (t0_samples * np.ones(len(a_p0_planets))) # warning, this also goes in the initial conditions when doing mulitple planets otherwise it won't work
    #initial conditions: both a_p0 and m0 take the outer planet and one of the inner planets
    sigma_gas_inner = sigma_gas_steady_state(a_p0_planets, t0_planets, params)
    m0_planets = M0_pla_Mstar(a_p0_planets, t0_planets, sigma_gas_inner, params)
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
