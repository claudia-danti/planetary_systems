
import cProfile #for checking the nr of calls and execution time
import pstats
from pstats import SortKey
import os
import numpy as np
import multiple_planets_gas_acc as code_gas
from functions_pebble_accretion import *
from functions import *
import functions_plotting as plot
import matplotlib.pyplot as plt
import matplotlib as mpl
import astropy.units as u
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
import sim_loader as sim_load

# Identify which simulation
folder_path = ["sims/gas_acc/stellar_masses/single_planets/liu"]

timestep = 10000
H_r_model = ['irradiated']

mstar = [0.1] # in M_sun
simulations = []
sim_parameters = []
parameters = []

for folder_path, H_r_model, mstar in zip(folder_path, H_r_model, mstar):
    # List all files in the given folder
    all_files = os.listdir(folder_path)
    # create the list of names of the sim, sim_params and params files 
    sim_filenames = [os.path.join(folder_path, f) for f in all_files if f.startswith('simulation_'+H_r_model) and f.endswith('N_steps'+str(timestep)+'_Mstar_'+str(mstar)+'.json')]
    sim_params_filenames = [os.path.join(folder_path, f) for f in all_files if f.startswith('sim_params_'+H_r_model) and f.endswith('N_steps'+str(timestep)+'_Mstar_'+str(mstar)+'.json')]
    params_filenames = [os.path.join(folder_path, f) for f in all_files if f.startswith('params_'+H_r_model) and f.endswith('N_steps'+str(timestep)+'_Mstar_'+str(mstar)+'.json')]

    # Sort the filenames based on the initial time
    sim_filenames.sort(key=sim_load.extract_initial_time)
    sim_params_filenames.sort(key=sim_load.extract_initial_time)
    params_filenames.sort(key=sim_load.extract_initial_time)

    # Load the simulations, sim_params, and params
    simulations.append([sim_load.JSONtoSimRes(filename) for filename in sim_filenames])
    sim_parameters.append([sim_load.load_sim_params(filename) for filename in sim_params_filenames])
    parameters.append([sim_load.load_params(filename) for filename in params_filenames])


fig, axs = plt.subplots(1,1, figsize=(8,6))
a_p0 = np.geomspace(5e-3, 1e2, num = 1000)

alpha_transp = 0.2
############### DISCS loops ########################

for j in range(len(simulations[0])):
    sim = simulations[0][j]
    params = parameters[0][j]
    sim_params = sim_parameters[0][j]
    for p in range(sim_params.nr_planets):

        idx = plot.idxs (axs, sim.time, sim.mass[p], sim.position[p], sim.filter_fraction[p], 
                        sim.dR_dt[p], sim.dM_dt[p], params, True)
        iso_idx = np.argmax(sim.mass[p] > M_peb_iso(sim.position[p].value, sim.time.value, params)*u.M_earth)
        stop_mig_idx = idx['stop_mig_idx'].values[0]
        cmap =  mpl.cm.inferno.reversed()
        norm = mpl.colors.LogNorm(vmin = sim_params.t_in, vmax = sim_params.t_fin)
        colors = cmap(norm(sim_params.t0[p]))

        axs.scatter(sim.position[p,0].to(u.au), sim.mass[p,0].to(u.M_earth), facecolors='none',edgecolors = colors, norm=norm, 
        cmap = cmap)

        if iso_idx != 0:
            axs.scatter(sim.position[p,iso_idx].to(u.au), sim.mass[p,iso_idx].to(u.M_earth),
                            color =  cmap(norm(sim.time[iso_idx].to(u.Myr).value)), facecolors='none', marker = 'v')

        axs.scatter(sim.position[p,stop_mig_idx].to(u.au), sim.mass[p,stop_mig_idx].to(u.M_earth), 
                    color =cmap(norm(sim.time[stop_mig_idx].to(u.Myr).value)))
        
        axs.scatter(sim.position[p, -1].to(u.au), Roman_Sensitivity(sim.position[p, -1].value))

axs.set_title("Irradiated, $M_{\star}=$"+str(params.star_mass*u.M_earth.to(u.M_sun))+" $M_{\odot}$", fontsize = 18)
plot.plot_roman_sensitivity(fig, axs)
axs.axvline(R_Einstein(8,4,params))
print("R_Einstein", R_Einstein(8,4,params))
axs.loglog(a_p0, M_peb_iso(a_p0, sim_params.t_in, params), color = "slateblue", linestyle =':', zorder = 0)
axs.loglog(a_p0, M_peb_iso(a_p0, sim_params.t_fin, params), color = "slateblue", linestyle =':', zorder = 0)
axs.fill_between(a_p0, M_peb_iso(a_p0, sim_params.t_fin, params), M_peb_iso(a_p0, sim_params.t_in, params),  color='slateblue', alpha=0.1)    #plot the magnetic cavity and shade the region inside it (from initial to final position)

#plot the initial mass line
from matplotlib import pyplot as plt, ticker as mticker
params = parameters[0][0]
sim_params = sim_parameters[0][0]

axs.axvline(r_magnetic_cavity(sim_params.t_in, params), linestyle = '-.', color = 'grey', alpha = 0.1)
axs.axvline(r_magnetic_cavity(sim_params.t_fin, params), linestyle = '-.', color = 'grey', alpha = 0.1)
axs.axvspan(r_magnetic_cavity(sim_params.t_in, params), r_magnetic_cavity(sim_params.t_fin, params),facecolor='none', hatch='/', edgecolor='gray', alpha =alpha_transp)

axs.set_xlabel('r [AU]', fontsize = 25, labelpad=20)
axs.set_xlim(5e-3, 1e2)
axs.tick_params(axis = "both", which = "major", direction = 'in', size = 15, labelsize = 18)
axs.tick_params(axis = "both", which = "minor", direction = 'in', size = 10)
axs.set_ylim(1e-9, 1e3)
axs.set_xscale('log')
axs.set_yscale('log')
plot.all_x_ticks(axs, num_ticks=100)
axs.set_ylabel('M [$M_{\oplus}$]', fontsize = 25, labelpad=20)

plt.savefig("figures/pop_synth/pop_synt_starmass_single_liu", bbox_inches='tight')


plt.show()
