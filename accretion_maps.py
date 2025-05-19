import numpy as np
import multiple_planets_gas_acc as code
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

color = mpl.colormaps["YlOrRd"].reversed()(np.linspace(0, 0.7, code.sim_params.nr_planets))


# params_dict = {'St_const': None, 
#                 'M_dot_star': None,
#                 #'M_dot_star': (1e-8*const.M_sun.cgs/u.yr).to(u.g/u.s),
#                 'iceline_radius': None,
#                 'alpha': 1e-2,
#                 'alpha_z': 1e-4, 
#                 'alpha_frag': 1e-4, 
#                 'iceline_alpha_change': False,
#                 'iceline_flux_change': False,
#                 'gas_accretion': False
#                 }
# t_initial = 0.1
# a_p0 = np.geomspace(50, 1e-1, num = 50)
# t0= ([t_initial] * np.ones(len(a_p0))) # warning, this also goes in the initial conditions when doing mulitple planets otherwise it won't work
# t_in = (t_initial) #is needed to start the simulation at the right time?
# sim_params_dict = {'N_step': 10000,
#                 'a_p0':a_p0,
#                 't0':t0,
#                 't_in':t_in
#                 }

# epsilon_el = [1, 1e-2, 1e-2]
# epsilon_h = [1, 0.5, 0.5]
# output_folder = 'sims/gas_acc/accretion_maps/models_5Myr'
# t_fin = 5 #Myr, end of sim
# N_steps = 10000 #number of steps of the sim 

# params_lam_ext = code.Params(**params_dict, H_r_model='Lambrechts_mixed', epsilon_el=epsilon_el[0], epsilon_heat=epsilon_h[0])
# params_lam = code.Params(**params_dict, H_r_model='Lambrechts_mixed', epsilon_el=epsilon_el[1], epsilon_heat=epsilon_h[1])
# params_irr = code.Params(**params_dict, H_r_model='irradiated', epsilon_el=epsilon_el[2], epsilon_heat=epsilon_h[2])
# peb_acc = code.PebbleAccretion(simplified_acc=False)
# gas_acc = code.GasAccretion()

# m0_lam_ext = M0_pla(a_p0, t_in, sigma_gas_steady_state(a_p0, t_in, params_lam_ext), params_lam_ext)
# sim_params_lam_ext = code.SimulationParams(**sim_params_dict, m0=m0_lam_ext)
# m0_lam = M0_pla(a_p0, t_in, sigma_gas_steady_state(a_p0, t_in, params_lam), params_lam)
# sim_params_lam = code.SimulationParams(**sim_params_dict, m0=m0_lam)
# m0_irr = M0_pla(a_p0, t_in, sigma_gas_steady_state(a_p0, t_in, params_irr), params_irr)
# sim_params_irr = code.SimulationParams(**sim_params_dict, m0=m0_irr)

# sim_lam_ext = code.simulate_euler(migration=False, filtering=False, peb_acc=peb_acc,gas_acc = gas_acc,  params=params_lam_ext, sim_params=sim_params_lam_ext, output_folder=output_folder)
# sim_lam = code.simulate_euler(migration=False, filtering=False, peb_acc=peb_acc, gas_acc = gas_acc, params=params_lam, sim_params=sim_params_lam, output_folder=output_folder)
# sim_irr = code.simulate_euler(migration=False, filtering=False, peb_acc=peb_acc, gas_acc = gas_acc, params=params_irr, sim_params=sim_params_irr, output_folder=output_folder)


params_dict = {'St_const': None, 
                'M_dot_star': None,
                'iceline_radius': None,
                'alpha': 1e-2,
                'alpha_z': 1e-4, 
                'alpha_frag': 1e-4, 
                'iceline_alpha_change': False,
                'iceline_flux_change': False,
                'gas_accretion': False
                }
t_initial = 0.3
t_final = 5
a_p0 = np.geomspace(50, 1e-1, num = 50)
t0= ([t_initial] * np.ones(len(a_p0))) # warning, this also goes in the initial conditions when doing mulitple planets otherwise it won't work
t_in = (t_initial) #is needed to start the simulation at the right time?
sim_params_dict = {'N_step': 10000,
                'a_p0':a_p0,
                't0':t0,
                't_in':t_in,
                't_fin':t_final
                }

mstar = np.array([0.1, 0.55, 1.0])*const.M_sun.to(u.M_earth).value

output_folder = 'sims/gas_acc/accretion_maps/stellar_masses_old'
t_fin = 5 #Myr, end of sim
N_steps = 10000 #number of steps of the sim 

params_irr1 = code.Params(**params_dict, H_r_model='irradiated', star_mass=mstar[1])
params_irr10 = code.Params(**params_dict, H_r_model='irradiated', star_mass=mstar[2])
peb_acc = code.PebbleAccretion(simplified_acc=False)
gas_acc = code.GasAccretion()

m0_irr1 = M0_pla_Mstar(a_p0, t_in, sigma_gas_steady_state(a_p0, t_in, params_irr1), params_irr1)
sim_params_irr1 = code.SimulationParams(**sim_params_dict, m0=m0_irr1)
m0_irr10 = M0_pla_Mstar(a_p0, t_in, sigma_gas_steady_state(a_p0, t_in, params_irr10), params_irr10)
sim_params_irr10 = code.SimulationParams(**sim_params_dict, m0=m0_irr10)

#sim_irr1 = code.simulate_euler(migration=False, filtering=False, peb_acc=peb_acc, gas_acc = gas_acc, params=params_irr1, sim_params=sim_params_irr1, output_folder=output_folder)
sim_irr10 = code.simulate_euler(migration=False, filtering=False, peb_acc=peb_acc, gas_acc = gas_acc, params=params_irr10, sim_params=sim_params_irr10, output_folder=output_folder)
