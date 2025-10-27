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

color = mpl.colormaps["YlOrRd"].reversed()(np.linspace(0, 0.7, code_gas.sim_params.nr_planets))


# disc parameters
params_dict = {'St_const': None, 
               'M_dot_star': None,
               'iceline_radius': None,
                'alpha': 1e-2,
                'alpha_z': 1e-4, 
                'alpha_frag': 1e-4, 
                'epsilon_el': 1e-2,
                'epsilon_heat':0.5,
                'v_frag': (1 * u.m/u.s).to(u.au/u.Myr).value,
                }
params = code_gas.Params(**params_dict, H_r_model='Lambrechts_mixed')



t_fin = 5 #Myr, end of sim
N_steps = 5000 #number of steps of the sim 

# Number of samples to generate
num_samples = 20
seed = 42
seed_t0 = 11
see_ap0 = 7

# inner embryo
t0_inner_samples = stats.uniform.rvs(0.1, 1, size=num_samples, random_state=seed_t0)
a_p0_inner_samples = stats.loguniform.rvs(1e-1, 1e1, size=num_samples, random_state=see_ap0)
t0_inner = ([t0_inner_samples] * np.ones(len(a_p0_inner_samples))) # warning, this also goes in the initial conditions when doing mulitple planets otherwise it won't work

# outer embryo
a_p0_outer = np.array([30])
t0_outer = ([0.2] * np.ones(len(a_p0_outer)) ) # warning, this also goes in the initial conditions when doing mulitple planets otherwise it won't work
m0_outer = M0_pla(a_p0_outer, t0_outer, sigma_gas_steady_state(a_p0_outer, t0_outer, params), params)

# number of processors
nprocs = 4 # change it if you want


all_parameters = [pair+(params,) for pair in zip(a_p0_inner_samples,t0_inner_samples)]

def kernel(params_tuple):
    try: 
        a_p0_in, t0_in, params = params_tuple
        #initial conditions: both a_p0 and m0 take the outer planet and one of the inner planets
        sigma_gas_inner = sigma_gas_steady_state(a_p0_in, t0_in, params)
        m0_in = M0_pla(a_p0_in, t0_in, sigma_gas_inner, params)
    
        a_p0 = np.array([*a_p0_outer, a_p0_in])
        m_0 = np.array([*m0_outer, m0_in])
        t0 = np.array([*t0_outer, t0_in])

        sim_params_dict = {'N_step': N_steps,
                        'm0': m_0,
                        'a_p0': a_p0,
                        't0': t0,
                        't_fin': t_fin,
                    }
        sim_params = code_gas.SimulationParams(**sim_params_dict)
        peb_acc = code_gas.PebbleAccretion(simplified_acc=False)
        gas_acc = peb.GasAccretion()

        result = code_gas.simulate_euler(migration = True, filtering = True, peb_acc = peb_acc, gas_acc=gas_acc, params=params, sim_params=sim_params)

        return True
    except:
        return False

# loop the sims
with mp.Pool(nprocs) as pool:
	all_results = pool.map(kernel, all_parameters)
	