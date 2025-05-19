import numpy as np
import astropy.units as u
import astropy.constants as const
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import sys
from functions import *
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.ticker
from matplotlib.ticker import ScalarFormatter, LogFormatter, LogLocator
from matplotlib import colors


########## GENERIC PLOTTING FUNCTION, FOR M(t), ff(t), GROWTH TRACKS #####################
# Don't require SimulationResults() objects

def idxs (axs, time, mass, position, filter_fraction, dR_dt, dM_dt, params, migration, **kwargs):
    #Creates the index dictionary

    idx_or_last = lambda fltr: np.argmax(fltr) if np.any(fltr) else fltr.size
    isolation_mass = M_peb_iso(position.value, time.value, params)
    stop_idx = np.argmin(position) #returns the position of the min value of position
    stop_mass_idx = np.any(np.where(dM_dt == 0)[0][0]) if np.any(np.where(dM_dt == 0)[0]) else dM_dt.size
    iso_idx = np.argmax(mass.value > isolation_mass)
    
    if iso_idx !=0:
        isolation_idx = iso_idx
    else:
        #print("iso index = mass size")
        isolation_idx = mass.size

    if migration:
        stop_mig_indices = np.where(dR_dt == 0)[0]
        if stop_mig_indices.size > 0:
            stop_mig_idx = stop_mig_indices[0]
        else:
            stop_mig_idx = len(dR_dt) - 1  # or some other default value
        # Returns the position of the first time for which r < r_mag
        #returns the position of the first time for which r<r_mag 
        inner_edge_idx = idx_or_last(position.value < r_magnetic_cavity(time.value, params))
        if stop_idx < inner_edge_idx:
            coll_or_res_idx = stop_idx
            end_idx = min(inner_edge_idx, coll_or_res_idx)

        else:
            coll_or_res_idx = None
            end_idx = inner_edge_idx

    else:
        
        # only because otherwise the dict complains
        inner_edge_idx = r_magnetic_cavity(time.value, params)
        coll_or_res_idx = -1
        end_idx = -1
        stop_mig_idx = -1
        #####
        if isolation_idx < mass.size:
            death_idx = isolation_idx
        else:
            death_idx = mass.size-1
        suicide_idx = death_idx

    if 1 in filter_fraction:
        saturation_idx = np.argmax(filter_fraction == 1)
    else:
        saturation_idx = None

    df = pd.DataFrame({
        'isolation_idx': [isolation_idx],
        'inner_edge_idx': [inner_edge_idx],
        'saturation_idx': [saturation_idx],
        'stop_idx': [stop_idx],
        'coll_or_res_idx': [coll_or_res_idx],
        'end_idx': [end_idx],
        'stop_mig_idx': [stop_mig_idx],
        'stop_mass_idx': [stop_mass_idx]
    })
    return df
"""
def idxs (axs, sim, params, sim_params, migration, **kwargs):
    #Creates the index dictionary

    idx_or_last = lambda fltr: np.argmax(fltr) if np.any(fltr) else fltr.size #returns the idx of the max value in vector if the vector has nonzero vaues else the size of the vector
    isolation_mass = M_peb_iso(sim.position, sim.time, params)
    
    isolation_idx = idx_or_last(sim.mass > isolation_mass)
    stop_idx = np.argmin(sim.position) #returns the position of the min value of position
    

    if migration:
        stop_mig_idx = np.where(sim.dR_dt[p].to(u.au/u.yr) == 0)[0][0] 
        #returns the position of the first time for which r<r_mag 
        inner_edge_idx = idx_or_last(sim.position < r_magnetic_cavity(sim.time, params))
        if stop_idx < inner_edge_idx:
            coll_or_res_idx = stop_idx
            end_idx = min(inner_edge_idx, coll_or_res_idx)

        else:
            coll_or_res_idx = None
            end_idx = inner_edge_idx
    else:
        if isolation_idx < sim.mass.size:
            death_idx = isolation_idx
        else:
            death_idx = sim.mass.size-1

    if 1 in sim.filter_fraction:
        saturation_idx = np.argmax(sim.filter_fraction == 1)
    else:
        saturation_idx = None


    df = pd.DataFrame({
        'isolation_idx': [isolation_idx],
        'inner_edge_idx': [inner_edge_idx],
        'saturation_idx': [saturation_idx],
        'stop_idx': [stop_idx],
        'coll_or_res_idx': [coll_or_res_idx],
        'end_idx': [end_idx],
        'stop_mig_idx': [stop_mig_idx]
    })
    
    return df
"""
def plot_mass(axs, sim, params, sim_params, migration, color, **kwargs):
    """Script to plot M vst t of the planets"""

    for p in range(sim_params.nr_planets-1,-1,-1):

        idx_df = idxs (axs, sim.time, sim.mass[p], sim.position[p], sim.filter_fraction[p], sim.dR_dt[p], params, migration, **kwargs)
        isolation_idx = idx_df['isolation_idx'].values[0]
        inner_edge_idx = idx_df['inner_edge_idx'].values[0]
        saturation_idx = idx_df['saturation_idx'].values[0]
        stop_idx = idx_df['stop_idx'].values[0]
        coll_or_res_idx = idx_df['coll_or_res_idx'].values[0]
        end_idx = idx_df['end_idx'].values[0]
        stop_mig_idx = idx_df['stop_mig_idx'].values[0]
        #end of simulation time

        axs.axvline(x = 1e7, linestyle="--", color = "grey", zorder = 0)
        # plotting of the M(t)
        axs.loglog(sim.time.to(u.yr)[p,:death_idx], sim.mass[p,:death_idx].to(u.earthMass), zorder=2, color = color, **kwargs)

        # plotting the isolation mass markers
        if isolation_idx < sim.mass.size:
            axs.scatter(sim.time.to(u.yr)[p,isolation_idx], sim.mass[p,isolation_idx].to(u.earthMass), color = 'black', zorder=2)
        # plotting the reached end of simulation markers
        else: 
            axs.scatter(sim.time.to(u.yr)[p,-1],sim.mass[p,-1].to(u.earthMass), marker = 'x', color = 'black', zorder=2)

        # saturation = reaching ff = 1 before pebble isolation mass
        if 1 in sim.filter_fraction:
            # checking that we reach 1 before isolation mass
            if saturation_idx < isolation_idx:
                axs.scatter(sim.time[p,saturation_idx].to(u.yr), (sim.mass[p,saturation_idx].to(u.earthMass)), marker = '*', color = color)

    all_y_ticks(axs, num_ticks= 100)

    axs.tick_params(axis = "both", length = 5, which = "minor")
    axs.tick_params(axis = "both", length = 7, which = "major")

    axs.set_ylabel("$M_c [M_{\oplus}]$", size = 15)
    axs.set_xlabel("t [yrs]", size = 15)
    axs.set_ylim(5e-11, 1e2)
    axs.set_xlim(8e4, 2e7)

def plot_filter_frac(axs, sim, params, sim_params, migration, **kwargs):
    """Plots the filtering fraction vs time of the simulated set of planets, possible migration"""
    for p in [-1]:
    #for p in range(sim_params.nr_planets-1,-1,-1):

        idx_df = idxs (axs, sim.time, sim.mass[p], sim.position[p], sim.filter_fraction[p], sim.dR_dt[p], sim.dM_dt[p], params, migration, **kwargs)
        isolation_idx = idx_df['isolation_idx'].values[0]
        inner_edge_idx = idx_df['inner_edge_idx'].values[0]
        saturation_idx = idx_df['saturation_idx'].values[0]
        stop_idx = idx_df['stop_idx'].values[0]
        coll_or_res_idx = idx_df['coll_or_res_idx'].values[0]
        end_idx = idx_df['end_idx'].values[0]
        stop_mig_idx = idx_df['stop_mig_idx'].values[0]

        if 1 in sim.filter_fraction[p]:
            # checking that we reach 1 before isolation mass
            if saturation_idx < isolation_idx:
                axs.scatter(sim.time[p,saturation_idx].to(u.yr), (sim.filter_fraction[p,saturation_idx])*100, marker = '*', **kwargs)

        if isolation_idx < sim.mass[p].size:
            axs.scatter(sim.time[isolation_idx+1].to(u.yr), (sim.filter_fraction[p,isolation_idx+1])*100, color = 'black', zorder =3)

        #mask = filter_fraction[i,:] != 0 # gets rid of the 0 values of the filter fraction once the planet dies in the star
        axs.plot(sim.time.to(u.yr), (sim.filter_fraction[p])*100, **kwargs)
    #axs.axvline(x = 1e7, linestyle="--", color = "grey", zorder = 0)
    axs.autoscale(enable=True, axis='x', tight=True)
    #axs.axhline(y = 50, color = "grey")
    #axs.axvspan((3*u.Myr).to(u.yr).value, (20*u.Myr).to(u.yr).value, alpha = 0.05, color = "grey")

    axs.tick_params(axis = "both", length = 5, which = "minor")
    axs.tick_params(axis = "both", length = 7, which = "major")
    axs.set_ylabel(" % Filtered fraction", size = 15)
    axs.set_xlabel("t [yrs]", size = 15)
    axs.set_xscale("log")
    axs.legend()
    axs.set_xlim(9e4, 1e7)


### WARNING: the isolation mass is the one at the final simulation, but beware that the viscously heated H/R depends on the time through M_dot
def plot_growth_track(axs, sim, params, sim_params, migration, color, add_ylabel=True, **kwargs):
    """Plots the mass vs time and the growth tracks of a system of planets"""

    for p in range(sim_params.nr_planets-1,-1,-1):

        idx_df = idxs (axs, sim.time, sim.mass[p], sim.position[p], sim.filter_fraction[p], sim.dR_dt[p], sim.dM_dt[p], params, migration, **kwargs)
        isolation_idx = idx_df['isolation_idx'].values[0]
        inner_edge_idx = idx_df['inner_edge_idx'].values[0]
        saturation_idx = idx_df['saturation_idx'].values[0]
        stop_idx = idx_df['stop_idx'].values[0]
        coll_or_res_idx = idx_df['coll_or_res_idx'].values[0]
        end_idx = idx_df['end_idx'].values[0]
        stop_mig_idx = idx_df['stop_mig_idx'].values[0]
        #plot the growth track
        axs.loglog(sim.position.to(u.au)[p,:isolation_idx], sim.mass[p,:isolation_idx].to(u.earthMass), color = color, zorder =1, **kwargs)

        #plot the initial mass line
        a_p0 = np.geomspace(0.03, 100, sim_params.nr_planets)
        m0 = M0_pla((a_p0*u.au).to(u.cm), sim_params.t0, sigma_gas_steady_state((a_p0*u.au).to(u.cm), sim_params.t0, params), params)
        axs.loglog(a_p0, m0.to(u.earthMass), linestyle  = ':', color = 'lightblue')
        #plot the isolation mass line
        axs.loglog(a_p0*u.au, M_peb_iso((a_p0*u.au).to(u.cm), sim_params.t_in, params).to(u.M_earth), color = "slateblue", linestyle =':')
        axs.loglog(a_p0*u.au, M_peb_iso((a_p0*u.au).to(u.cm), sim_params.t_fin, params).to(u.M_earth), color = "slateblue", linestyle =':')

        #if isolation_idx < sim.mass.size:
            #axs.scatter(sim.position.to(u.au)[p,isolation_idx+1], sim.mass[p,isolation_idx+1].to(u.earthMass), color = 'black', zorder = 2)
        #else: 
            #axs.scatter(sim.position.to(u.au)[p,-1],sim.mass[p,-1].to(u.earthMass), marker = 'x', color = 'black', zorder=2)

        if 1 in sim.filter_fraction:
            axs.scatter(sim.position[p,saturation_idx].to(u.au), (sim.mass[p,saturation_idx].to(u.earthMass)), marker = '*', color = color)
        

        axs.tick_params(axis = "both", length = 15)
        axs.tick_params(axis = "both", length = 10, which = "minor")
        if add_ylabel:
            axs.set_ylabel("$M \: [M_{\oplus}]$", size = 20) 
        axs.set_xlabel("r [AU]", size = 20) 
        axs.set_ylim(1e-7, 1e2)
        axs.set_xlim(3e-2, 1e2)

        all_y_ticks(axs, num_ticks=100)


def plot_growth_track_timescale(fig, axs, sim, params, sim_params,  migration, cmap, add_ylabel=True, add_cbar=True, nofilter=False, **kwargs):
    """Plots the mass vs time and the growth tracks of a system of planets"""

    for p in range(sim_params.nr_planets-1,-1,-1):

        idx_df = idxs (axs, sim.time, sim.mass[p], sim.position[p], sim.filter_fraction[p], sim.dR_dt[p], sim.dM_dt[p], params, migration, **kwargs)
        isolation_idx = idx_df['isolation_idx'].values[0]
        inner_edge_idx = idx_df['inner_edge_idx'].values[0]
        saturation_idx = idx_df['saturation_idx'].values[0]
        stop_idx = idx_df['stop_idx'].values[0]
        coll_or_res_idx = idx_df['coll_or_res_idx'].values[0]
        end_idx = idx_df['end_idx'].values[0]
        stop_mig_idx = idx_df['stop_mig_idx'].values[0]
        stop_mass_idx = idx_df['stop_mass_idx'].values[0]
        pos = np.geomspace(1e-2,200, num=sim_params.N_step+1)
        norm=mpl.colors.LogNorm(vmin = sim_params.t_in, vmax = sim_params.t_fin)
        print('planet '+str(sim_params.a_p0[p]*(u.au))+" iso mass at ", sim.time[isolation_idx].to(u.Myr) if isolation_idx < sim.mass[p].size else "no iso, end of sim")
        #print('inner', inner_edge_idx)
        #plot the growth track with color coding gven by the time it takes to grow
        #scatter grey points after they reach isolation mass

        if nofilter:

            sc_post_iso = axs.loglog(sim.position.to(u.au)[p,isolation_idx:stop_mig_idx], sim.mass[p,isolation_idx:stop_mig_idx].to(u.earthMass), color='grey', linewidth=6, alpha = 0.1)
            axs.scatter(sim.position[p,stop_mig_idx].to(u.au), sim.mass[p,stop_mig_idx].to(u.earthMass), marker = 'x', color = 'grey', s = 100, zorder=100)
            if p!=0:
                axs.loglog(sim.position.to(u.au)[p,isolation_idx:stop_mig_idx], sim.mass[p,isolation_idx:stop_mig_idx].to(u.earthMass), color='white', linestyle =':', linewidth=9)

            if isolation_idx < sim.mass[p].size:
                dt= (sim.time[:isolation_idx]).to(u.Myr)
                sc = axs.scatter(sim.position.to(u.au)[p,:isolation_idx], sim.mass[p,:isolation_idx].to(u.earthMass), c=dt, norm=norm,  cmap = cmap)
                axs.scatter(sim.position.to(u.au)[p,isolation_idx], sim.mass[p,isolation_idx].to(u.earthMass), color = 'black', s = 150, zorder = 100)
            else: 
                dt= (sim.time[:stop_mig_idx]).to(u.Myr)
                sc = axs.scatter(sim.position.to(u.au)[p,:stop_mig_idx], sim.mass[p,:stop_mig_idx].to(u.earthMass), c=dt, norm=norm,  cmap = cmap)
            
            axs.loglog(sim.position.to(u.au)[p,:isolation_idx], sim.mass[p,:isolation_idx].to(u.earthMass), color='white', linestyle =':', linewidth=9, zorder =1)

        else:
            if isolation_idx < sim.mass[p].size:
                dt= (sim.time[:isolation_idx]).to(u.Myr)
                sc = axs.scatter(sim.position.to(u.au)[p,:isolation_idx], sim.mass[p,:isolation_idx].to(u.earthMass), c=dt, norm=norm,  cmap = cmap)
                axs.scatter(sim.position.to(u.au)[p,isolation_idx], sim.mass[p,isolation_idx].to(u.earthMass), color = 'black', s = 150, zorder = 100)
            else: 
                dt= (sim.time[:stop_mig_idx]).to(u.Myr)
                sc = axs.scatter(sim.position.to(u.au)[p,:stop_mig_idx], sim.mass[p,:stop_mig_idx].to(u.earthMass), c=dt, norm=norm,  cmap = cmap)

                #axs.scatter(sim.position.to(u.au)[-1],sim.mass[-1].to(u.earthMass), marker = 'x', color = 'grey',s = 100, zorder=100)
            axs.scatter(sim.position[p,stop_mig_idx].to(u.au), sim.mass[p,stop_mig_idx].to(u.earthMass), marker = 'x', color = 'grey', s = 100, zorder=100)
            sc_post_iso = axs.loglog(sim.position.to(u.au)[p,isolation_idx:stop_mig_idx], sim.mass[p,isolation_idx:stop_mig_idx].to(u.earthMass), color='grey', linewidth=6, alpha = 0.1)

            #if 1 in sim.filter_fraction:
                #axs.scatter(sim.position[p,saturation_idx].to(u.au), (sim.mass[p,saturation_idx].to(u.earthMass)), marker = '*', color = 'black')

             

    if add_cbar:
        #handling the colorbar	
        #fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(sc, cax=cbar_ax)	
        # Manually set the colorbar boundaries and ticks
        cbar_ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0, 2.0, 5.0, ]))
        cbar_ax.yaxis.set_major_formatter(LogFormatter())
        cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(custom_log_formatter))
        cbar.set_label('accretion time [Myr]', fontsize=25, labelpad=15)
        cbar.ax.tick_params(axis = 'both', which = 'major', size = 18, labelsize = 18)
        cbar.ax.tick_params(axis = 'both', which = 'minor', size = 12)
        
   
    # #plot the initial mass line
    print(sim_params.t_in)
    a_p0 = np.geomspace(1e-3, 1e2, num = 1000)
    m0 = M0_pla_Mstar(a_p0, sim_params.t_in, sigma_gas_steady_state(a_p0, sim_params.t_in, params), params)
    axs.loglog(a_p0, m0, linestyle  = ':', color = 'lightblue', zorder = 0)
    #plot the isolation mass line
    axs.loglog(a_p0, M_peb_iso(a_p0, sim_params.t_in, params), color = "slateblue", linestyle =':', zorder = 0)
    axs.loglog(a_p0, M_peb_iso(a_p0, sim_params.t_fin, params), color = "slateblue", linestyle =':', zorder = 0)

    #plot the magnetic cavity and shade the region inside it (from initial to final position)
    axs.axvline(r_magnetic_cavity(sim_params.t_in, params), linestyle = '-.', color = 'grey', alpha = 0.05)
    axs.axvline(r_magnetic_cavity(sim_params.t_fin, params), linestyle = '-.', color = 'grey', alpha = 0.05)
    axs.axvspan(r_magnetic_cavity(sim_params.t_in, params), r_magnetic_cavity(sim_params.t_fin, params),facecolor='none', hatch='/', edgecolor='gray', alpha =0.05)

    axs.tick_params(axis = "both", length = 15, labelsize = 18)
    axs.tick_params(axis = "both", length = 10, which = "minor", labelsize = 18)
    if add_ylabel:
        axs.set_ylabel("$M \: [M_{\oplus}]$", size = 25) 
    axs.set_xlabel("r [AU]", size = 25) 
    axs.set_ylim(1e-9, 7e2)
    #axs.set_xlim(5e-3, 1e2)
    axs.set_xlim(1e-2, 2)

    all_y_ticks(axs, num_ticks=100)
    all_x_ticks(axs, num_ticks=100)




######################## COMBINED PLOTTING FUNCTIONS ###############
# Generally require SimulationResults() objects

def plot_mass_filter_frac(axs, sim, sim_params, params, color, migration, multiple_sims_idx = 0, title = None, img_name=None, **kwargs):

    # for the M(t) plot legend
    marker_list = ["D", "s", "X", "v"]
    labels, t_values,  m_values = []*sim_params.nr_planets,[]*sim_params.nr_planets,[]*sim_params.nr_planets
    
    axs[0].axvline(x = 2e6, linestyle="--", color = "lightgrey")


    for i in range(sim_params.nr_planets-1,-1,-1):

        axs[0].axhline(y = (M_peb_iso(sim.position[i,-1], params).to(u.M_earth)).value, color = color[i], linestyle ='--')
        # left axis has the M(t) and the pebble isolation mass lines
        plot_mass(sim.time, sim.mass[i,:], sim.position[i],sim.filter_fraction[i], params,
                    axs = axs[0], color = color[i], migration=migration, **kwargs)
        # ACCRETION REGIME TRANSITIONS
        # i is the key for the planet
        planet_times = sim.acc_regimes.get(i, {}).get("times", {})
        planet_masses = sim.acc_regimes.get(i, {}).get("masses", {})
        # Extract times in years and mass in M_earth of the accretion regime transition
        t_values = [(time).to(u.yr) for time in planet_times.values() ]
        labels = list(planet_times.keys()) # labels of the accretion regimes (they are the same in times and masses)
        m_values = [(mass).to(u.M_earth) for mass in planet_masses.values()]
        
        for j in range(len(t_values)):
            if multiple_sims_idx == 0:
                label = labels[j] if i == 0 else None
                label_F0 = '% of $F_0$ on inner planet' if i == 0 else None
            else:
                label = None
                label_F0 = None

            axs[0].scatter(t_values[j], m_values[j], label = label if i == 0 else None, color = color[i], marker = marker_list[j])

        # right axis has the ff(t)
        plot_filter_frac(sim.time, sim.filter_fraction[i], sim.mass[i], sim.position[i], params, migration, axs = axs[1], color = color[i], zorder = 0, **kwargs)
    
    axs[1].semilogx(sim.time.to(u.yr), (sim.flux_ratio[-1])*100, label = label, color= "navy", zorder = 1)
    #axs[1].semilogx(sim.time.to(u.yr), (sim.flux_on_planet[-1]/sim.F0[-1])*100, label = label_F0, color= "navy", zorder = 1)
    axs[0].axvspan((3*u.Myr).to(u.yr).value, (20*u.Myr).to(u.yr).value, alpha = 0.005, color = "grey")
    axs[1].axvspan((3*u.Myr).to(u.yr).value, (20*u.Myr).to(u.yr).value, alpha = 0.005, color = "grey")

    # legends
    marker_list = ["o", "*", "X"]
    label_list = ["peb iso", "saturation lim", "end sim"]
    marker_dim = [7, 10, 8]
    legend_handler(axs[1],"center left", marker_list, label_list, marker_dim)
    axs[0].legend(loc="upper left")

    axs[0].set_ylim(5e-6, 1e2)

    axs[1].set_title("Filter fraction of non-mig system, St = "+str(params.St_const))
    axs[0].set_title("Planetary masses of non-mig system, St = "+str(params.St_const))
    if title:
        plt.suptitle(title)
    if img_name:
        plt.savefig("figures/series_plots/St_"+str(params.St_const)+"_"+str(img_name)+".png")


def plot_mass_growth_track(axs, sim, params, sim_params, color, migration, multiple_sims_idx = 0, title = None, img_name=None, **kwargs):
    """Plots the mass vs time (left plot) and the growth tracks (right plot) of a system of planets, for one sim"""
    
    # PANEL 1: MASS EVOLUTION
    marker_list1 = ["D", "s", "P", "v"]

    labels, t_values, condition_indices, m_values = []*sim_params.nr_planets,[]*sim_params.nr_planets,[]*sim_params.nr_planets,[]*sim_params.nr_planets

    for i in range(sim_params.nr_planets-1,-1,-1):
        # mark the transition of the accretion regimes

        #`i` is the key for the planet
        planet_times = sim.acc_regimes.get(i, {}).get("times", {})
        planet_masses = sim.acc_regimes.get(i, {}).get("masses", {})

        # Extract non-zero times in years
        t_values = [(time).to(u.yr) for time in planet_times.values() ]
        labels = list(planet_times.keys())
        m_values = [(mass).to(u.M_earth) for mass in planet_masses.values()]

        for j in range(len(t_values)):
            
            if multiple_sims_idx == 0:
                label = labels[j] if i == 0 else None
            else:
                label = None
            
            axs[0].scatter(t_values[j], m_values[j], label = None, color = color[i], marker = marker_list1[j], zorder = 2)

        plot_mass(sim.time, sim.mass[i],  sim.position[i], sim.filter_fraction[i], params, axs[0], migration, color = color[i], label = None )
        plot_growth_track(sim.time, sim.filter_fraction[i], sim.mass[i], sim.position[i], migration, params, axs[1], color = color[i])

    axs[0].axvline(x = 2e6, linestyle="--", color = "lightgrey", zorder = 0)
    axs[1].axvline(x = 5, linestyle="--", color = "lightgrey", zorder = 0)
    axs[1].axvline(x = params.iceline_radius.to(u.au).value, linestyle="--", color = "lightblue", zorder = 0, label = "ice line")

    axs[0].legend()
    label_list1 = ["3D accretion", "Bondi accretion", "2D accretion", "Hill accretion"]
    marker_dim = [8,8,8,8]

    legend_handler(axs[0],"lower right", marker_list1, label_list1, marker_dim)


    axs[1].legend()
    marker_list = ["o", "*", "X"]
    label_list = ["peb iso", "saturation lim", "end sim"]
    marker_dim = [7, 10, 8]
    legend_handler(axs[1],"lower right", marker_list, label_list, marker_dim)

    plt.tight_layout()

    if title:
        plt.suptitle(title)
    if img_name:
        plt.savefig("figures/mass_gt/St_"+str(params.St_const)+"_"+str(img_name)+".png")


def plot_mass_filter_frac_growth_track(simulation, sim_params, params, color, axs1, axs2, axs3, migration, multiple_sims_idx = 0, title = None, img_name=None, **kwargs):
    """Plots M(t), ff(t), growth tracks, requires the use of gridspec"""
    """This needs to be used before calling the function
    fig = plt.figure(figsize=(12, 15))
    gs = gridspec.GridSpec(2, 4)
    axs1 = plt.subplot(gs[0, :2])
    axs2 = plt.subplot(gs[0, 2:])
    axs3 = plt.subplot(gs[1, 1:3])"""

    # PANEL 1: MASS EVOLUTION
    marker_list = ["D", "s", "P", "v"]

    labels, t_values, condition_indices, m_values = []*sim_params.nr_planets,[]*sim_params.nr_planets,[]*sim_params.nr_planets,[]*sim_params.nr_planets

    for i in range(sim_params.nr_planets-1,-1,-1):
        # mark the transition of the accretion regimes

        #`i` is the key for the planet
        planet_times = simulation.acc_regimes.get(i, {}).get("times", {})
        planet_masses = simulation.acc_regimes.get(i, {}).get("masses", {})

        # Extract non-zero times in years
        t_values = [(time).to(u.yr) for time in planet_times.values() ]
        labels = list(planet_times.keys())
        m_values = [(mass).to(u.M_earth) for mass in planet_masses.values()]
        for j in range(len(t_values)):
            if multiple_sims_idx == 0:
                label = labels[j] if i == 0 else None
            else:
                label = None
            axs1.scatter(t_values[j], m_values[j], label = label, color = color[i], marker = marker_list[j], zorder = 2)
        
        #axs1.axhline(y = (M_peb_iso(simulation.position[i,-1]).to(u.M_earth)).value, color = color[i], linestyle ='--', label = "Peb iso value" if i == 3 else None)
        #label_mass = str((sim_params.a_p0[i]).to(u.au))+ " planet"
        plot_mass(simulation.time, simulation.mass[i,:], simulation.position[i],simulation.filter_fraction[i], params, 
                    axs = axs1, label= None, color = color[i], migration = False)

    axs1.axvline(2e6, color = "grey", linestyle = "--", alpha = 0.5)    
    axs1.set_title("Planetary masses of non-mig system, St = "+str(params.St_const))
    #axs1.set_ylim(5e-8, 1e3)
    #axs1.set_ylim(1e-5, 1e3)
    axs1.set_xlim(8e4, 2e7)
    axs1.legend(loc="lower left")

    all_y_ticks(axs1, num_ticks=100)

    # PANEL 2: FILTER FRACTIONS

    if multiple_sims_idx == 0:
        label = '% of $F_0$ on inner planet'
        label2 = "50% flux reduction"
    else:
        label = None
        label2 = None

    for i in range(sim_params.nr_planets-1,-1,-1):

        plot_filter_frac(simulation.time, simulation.filter_fraction[i], simulation.mass[i], simulation.position[i], params, migration = False, axs = axs2, label = None, color = color[i], zorder = 0)

    axs2.semilogx(simulation.time.to(u.yr), (simulation.flux_ratio[-1])*100, label = label, color= "steelblue", zorder = 1)

    axs2.set_title("Filter fraction of non-mig system, St = "+str(params.St_const))
    axs2.axvline(x = 1e7, linestyle="--", color = "grey", label = "end time simulation" if j ==0 else None, zorder = 0)
    axs2.set_xlim(1e5, 2e7)
    axs2.legend(loc="center right")


    marker_list = ["o", "*", "X"]
    label_list = ["peb iso", "saturation lim", "end sim"]
    marker_dim = [7, 10, 8]
    legend_handler(axs2,"center right", marker_list, label_list, marker_dim)

    # PANEL 3: GROWTH TRACKS
    for i in range(sim_params.nr_planets):
        if multiple_sims_idx == 0 and i == 0:
            label =None
            #label = str((sim_params.a_p0[0]).to(u.au))+ " (outer) planet" 
        else:
            label = None
        plot_growth_track(simulation.time[-1], simulation.filter_fraction[i,:], simulation.mass[i,:], simulation.position[i,:], migration, params, sim_params,
                    axs=axs3, label= label, color = color[i])

    axs3.autoscale(enable=True, axis='x', tight=True)
    axs3.set_xlabel("R[AU]", size = 15)
    if params.iceline_radius == None:
        ## this is needed for plotting purposes, cause irr iceline is not time dependent thus not a vector
        if params.H_r_model == 'irradiated':
            axs3.loglog(np.ones_like(simulation.time).value*iceline(simulation.time,170*u.K, params).to(u.au), simulation.mass[-1,:].to(u.earthMass),  color = 'lightblue')
        else:
            axs3.loglog(iceline(simulation.time, 170*u.K, params).to(u.au), simulation.mass[-1,:].to(u.earthMass),  color = 'lightblue')
    else:
        axs3.axvline(x = params.iceline_radius.to(u.au).value, linestyle="--", color = "lightblue", zorder = 0)
    axs3.axvline(x = 1, linestyle="--", color = "grey", zorder = 0)
    axs3.axvline(x = 5, linestyle="--", color = "grey", zorder = 0)
    axs3.axhline(y = 1, linestyle='--', color='lightgrey', zorder =0)

    #axs3.set_xlim(1e-2, 2e2)
    #axs3.set_ylim(1e-5, 1e2)
    axs3.set_ylim(5e-8, 2e2)
    axs3.set_title("Growth tracks")

    all_y_ticks(axs3, num_ticks=100)

    plt.tight_layout()
  
    if title:
        plt.suptitle(title)
    if img_name:
        plt.savefig("figures/mass_ff_gt/St_"+str(params.St_const)+"_"+str(img_name)+".png")


def plot_flux(times, time_vector, flux, position, R_in, filter_frac, params, line_times):
    """"Plot of the flux as a function of the position for different time snapshots"""
    #times = time snapshots at which I want the flux to be plotted
    # time_vector = sim.times (simulation time vector)
    # flux - sim.flux_on_planets
    # position = sim.positions
    # R_in
    # filter_frac = sim.filter_fraction
    # line_times = 

    fig, axs = plt.subplots(1,1, figsize = (6,6))

    color = color = mpl.colormaps["inferno"].reversed()(np.linspace(0.1, 0.7, len(times)))

    position_list, flux_list = [], []
    for i,t in enumerate(times):
        idx = np.searchsorted(time_vector.to(u.s), t.to(u.s)) # finds the index in the vector where the time is = times
        positions = np.hstack([ position[:,idx].to(u.au), R_in * u.au]) # creates position vector by appending
        #F0 = flux_peb_t(t, position, params).cgs.to(u.M_earth/u.Myr)
        F_inside = flux[-1,idx].to(u.M_earth/u.Myr)*(1-filter_frac[-1,idx]) # computes the flux inside inner planet
        fluxes = np.hstack([ flux[:,idx].to(u.M_earth/u.Myr), F_inside ]) # creates flux vector by appending the initial flux for the outer disc, the inner flux for the inner part

        if t in line_times:
            axs.step(positions, fluxes.to(u.earthMass/u.Myr), where='pre', label=f"{t.to(u.Myr):.1f}", color = color[i])
            axs.scatter(positions[:-1], fluxes[:-1].to(u.earthMass/u.Myr), color = color[i])
        position_list.append(positions[1:-1])
        flux_list.append(fluxes[1:-1])
    
    position_list, flux_list = np.array(position_list), np.array(flux_list)
    # plotting  vertical grey dashed lines for the planet positions
    #for pos in position[:,idx].to(u.au):
        #axs.axvline(x = pos.value, color = 'grey', alpha=0.4)
    
    axs.axvline(x=2, color = 'lightblue', alpha = 0.5, linestyle = '--')
    axs.plot(position_list, (flux_list*u.g/u.s).to(u.earthMass/u.Myr), color = 'grey', alpha=0.4, linestyle =  '--')
    st = params.St_const
    axs.set_xlabel("R [AU]", size = 15)
    axs.set_ylabel("F [$M_{\oplus}$/Myrs]", size = 15)
    axs.set_title("Flux past each planet, St = "+str(st))
    axs.tick_params(axis = "both", which = "both", length = 8)
    axs.legend(loc = "lower right")#bbox_to_anchor=(0.7,0.6))

    plt.savefig("figures/tests/fluxes_St_"+str(st)+".png")


def plot_growth_tracks_ff_transition_reg(axs, sigma_gas, sim, params, sim_params, img_name):
    """Plots the growth tracks of the planets, color coded the accretion efficiency, overplotting the transition between accretion regimes"""
    
    acc_regimes_lines(axs, sim, params, sim_params)


    axs.axvline(x = 2, linestyle="--", color = "lightblue", alpha = 1, zorder = 0)
    axs.axvline(x = 1, linestyle="--", color = "grey", alpha = 0.3, zorder = 0)
    axs.axvline(x = 5, linestyle="--", color = "grey", alpha = 0.3, zorder = 0)

    all_y_ticks(axs, num_ticks=100)

    ##### ACCRETION EFFICIENCY CALCULATION
    sc = axs.scatter(sim.position.to(u.au),  sim.mass.to(u.earthMass), c=sim.filter_fraction,  norm=mpl.colors.LogNorm(), cmap = "viridis")
    cbar = plt.colorbar(sc)
    cbar.set_label(label="Accretion efficiency", size=15)
    cbar.ax.tick_params(labelsize=12)

    axs.axvspan(0.03, 2e-3, alpha = 0.1, color = "grey")

    #axs.set_ylim(1e-4, 1e3)
    axs.set_xlim(1e-2, 2e2)

    axs.tick_params(axis = "both", length = 8)
    axs.tick_params(axis = "both", length = 5, which = "minor")
    axs.set_xlabel("R [AU]", size = 15)
    axs.set_ylabel("$\mathrm{M[M_{\oplus}]}$", size = 15)
    axs.set_title("Planetary growth tracks, $v_{\mathrm{frag}}$="+str(params.v_frag.to(u.m/u.s)), size = 15)
    #axs.legend()
    plt.tight_layout()


    plt.savefig("figures/gt_acc_reg/growth_tracks_accretion_efficiency"+str(img_name)+".png")


def growth_timescale_heatmap(sim, params, sim_params, fig, axs,  title, fig_name, add_ylabel=True, add_cbar = True):
    num = 10000
    # binning the mass on the y-axis between the min and max of the starting and final masses
    M_in_min = M0_pla(sim.position[:,0], sim_params.t_in, sigma_gas_steady_state(sim.position[:,0], sim_params.t_in, params), params).min()
    M_fin_max = M_peb_iso(sim.position[:,0], sim_params.t_in, params).max() #the iso mass decreases with time in the viscous case
    m_bins = np.geomspace(M_in_min, M_fin_max, num = num)
    #print("mass binning", m_bins.to(u.M_earth))
    Z = np.zeros((len(m_bins), sim_params.nr_planets))*u.Myr # to be filled with accretion timescales

    for p in range(sim_params.nr_planets): #loop over the nr_planets
        for m in range(len(m_bins)): #loops over the mass of the planet (timesteps)
            t_idx = np.argmax(sim.mass[p].to(u.M_earth) >= m_bins[m])
            Z[m,p] = sim.time[t_idx].to(u.Myr)
    #axs.loglog(sim.position[:,0].to(u.au), M_peb_iso(sim.position[:,0], sim_params.t_in, params).to(u.M_earth), 
            #label = "M_peb_iso", color = 'slateblue', linestyle = ':') #caveat-> this is the M_iso at initial time, for the viscous case it decreases with time
    axs.loglog(sim.position[:,0].to(u.au), M_peb_iso(sim.position[:,0], sim_params.t_fin, params).to(u.M_earth), 
            label = "M_peb_iso", color = 'slateblue', linestyle = ':') #caveat-> this is the M_iso at final time

    axs.loglog(sim.position[:,0].to(u.au), M0_pla(sim.position[:,0], sim_params.t_in, 
                                                sigma_gas_steady_state(sim.position[:,0], sim_params.t_in, params), params).to(u.M_earth), 
                                                label = "M_in", color = 'lightblue', linestyle = ':')
    #axs.loglog(sim.position[:,0].to(u.au), M_3D_2DH_trans(sim.position[:,0].to(u.au), sim_params.t_in, params).to(u.M_earth).value, color = 'grey', linestyle = ':') #caveat, this is the transition mass for t_in, for viscous case it is time dependent
    #axs.loglog(sim.position[:,0].to(u.au), M_3D_2DH_trans(sim.position[:,0].to(u.au), sim_params.t_fin, params).to(u.M_earth).value, color = 'black', linestyle = ':') #caveat, this is the transition mass for t_in, for viscous case it is time dependent

    X,Y = np.meshgrid(sim.position[:,0].to(u.au), m_bins.to(u.M_earth))
    levs = np.geomspace(Z.min(), Z.max(), num = 20)
    print(levs)
    #plot white for the regions where the planet is dead or non existent
    Z[Z.value<=1e-1] = np.nan
    cmap = mpl.colormaps["inferno"].reversed()
    cmap.set_bad(color='white')
    masked_data = np.ma.masked_invalid(Z.value)
    print(masked_data)
    CS = axs.contourf(X, Y, masked_data, levels= levs, norm=colors.LogNorm(), ticks=LogLocator(subs='all'), boundaries=levs, cmap = cmap)

    if add_cbar:
        #handling the colorbar	
        #fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.95, 0.15, 0.03, 0.7])
        cbar = fig.colorbar(CS, cax=cbar_ax)	


         # Manually set the colorbar boundaries and ticks
        cbar_ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0, 5.0]))
        cbar_ax.yaxis.set_major_formatter(LogFormatter())
        cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(custom_log_formatter))
        cbar.set_label('accretion timescale [Myr]', fontsize=20, labelpad=15)
        cbar.ax.tick_params(axis = 'both', which = 'major', size = 10)
        cbar.ax.tick_params(axis = 'both', which = 'minor', size = 5)


    # Set the title and axis labels
    axs.set_title(title, fontsize = 20)
    axs.set_ylim(M_in_min.to(u.M_earth).value, M_fin_max.to(u.M_earth).value)
    axs.set_xlabel('r [AU]', fontsize = 25, labelpad=20)
    if add_ylabel:
        axs.set_ylabel('M [$M_{\oplus}$]', fontsize = 25, labelpad=20)
    axs.tick_params(axis='both', which='major', size=15)
    axs.tick_params(axis='both', which='minor', size=10)
    axs.tick_params(axis='both', which='major', labelsize=15)
    axs.figure.axes[-1].yaxis.label.set_size(22)
    axs.figure.axes[-1].tick_params(axis='both', which='minor', labelsize=10)
    axs.figure.axes[-1].tick_params(axis='both', which='major', labelsize=15)
    all_y_ticks(axs, 100)
    axs.set_ylim(1e-6, 1e2)
    #plt.savefig("figures/tests/"+fig_name, bbox_inches='tight')

def custom_log_formatter_old(x, pos):
	if x == 0:
		return "0"
	exponent = int(np.floor(np.log10(x)))
	coeff = x / 10**exponent
	if coeff == 1:
		return r"$10^{{{}}}$".format(exponent)
	else:
		return  r"${:.1f} \times 10^{{{}}}$".format(coeff, exponent)

def custom_log_formatter(x, pos):
    exponent = int(np.floor(np.log10(x)))
    coeff = x / 10**exponent
    if x == 0:
        return "0"
    else:   
        return x



def generate_levels(Z):
	Zmin = np.min(Z)
	Zmax = np.max(Z)
	
	# Find the closest power of 10 below Zmin
	start = 10**np.floor(np.log10(Zmin))
	
	# Generate levels: powers of 10 and 5*10^n
	levels = []
	while start <= Zmax:
		levels.append(start)
		levels.append(5 * start)
		start *= 10
	
	return levels

def legend_handler(axs, legend_location, marker_list, label_list, marker_dim):
    """Creates a custum legend (not based on the one when you plot stuff)"""

    custom_legend_handles = []
    for i in range(len(marker_list)):
        custom_legend_handles.append(plt.Line2D([0], [0], marker=marker_list[i], color='w', markerfacecolor='black', markersize=marker_dim[i], label=label_list[i]))

    # Get the automatic legend and its handles/labels
    automatic_legend = axs.get_legend()
    automatic_legend_handles, automatic_legend_labels = automatic_legend.legendHandles, [t.get_text() for t in automatic_legend.get_texts()]

    # Manually create a legend by combining automatic and custom legend entries
    legend_handles = automatic_legend_handles + custom_legend_handles
    legend_labels = automatic_legend_labels + [handle.get_label() for handle in custom_legend_handles]

    # Create the combined legend
    legend = axs.legend(handles=legend_handles, labels=legend_labels, loc=legend_location)


def create_acc_reg_dataframe(planet_id, planet_data):
    """Creates the accretion dictionary"""

    masses_data = {key: round(value.value, 10) for key, value in planet_data['masses'].items()}
    times_data = {key: round(value.value, 10) for key, value in planet_data['times'].items()}
    
    df = pd.DataFrame({
        'time [Myr]': times_data.values(),
        'mass [Earth]': masses_data.values(),
        'accretion_type': masses_data.keys(),
        'planet': planet_id
    })
    
    return df

def display_acc_df(acc_reg_dict):
    """Displays the accretion dictionary"""

    # Create DataFrames for each planet'
    planet_dfs = [create_acc_reg_dataframe(planet_id, planet_data) for planet_id, planet_data in acc_reg_dict.items()]

    # Display the DataFrames
    for planet_df in planet_dfs:
        print(planet_df)
        print()

def all_y_ticks(axs, num_ticks):
    """to plot all the ticks on the y axis, N.B. num_ticks must be bigger than the nr big ticks"""

    y_major = mpl.ticker.LogLocator(base = 10.0, numticks = num_ticks)
    axs.yaxis.set_major_locator(y_major)
    y_minor = mpl.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = num_ticks)
    axs.yaxis.set_minor_locator(y_minor)
    axs.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    plt.tick_params(axis = 'both', which = 'major', size = 10)
    plt.tick_params(axis = 'both', which = 'minor', size = 5)

def all_x_ticks(axs, num_ticks):
    """to plot all the ticks on the y axis, N.B. num_ticks must be bigger than the nr big ticks"""

    x_major = mpl.ticker.LogLocator(base = 10.0, numticks = num_ticks)
    axs.xaxis.set_major_locator(x_major)
    x_minor = mpl.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = num_ticks)
    axs.xaxis.set_minor_locator(x_minor)
    axs.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    plt.tick_params(axis = 'both', which = 'major', size = 10)
    plt.tick_params(axis = 'both', which = 'minor', size = 5)


def acc_regimes_lines(axs, sim, params, sim_params):
    """To label on the lines the accretion regimes"""

    # sampling of the space to get the transiton masses
    pos = (np.geomspace(0.001, 1000, num=100)*u.au).to(u.cm)
    #mass =  M0_pla(sim_params.a_p0, sigma_gas, params)

    stokes_peb = st_frag(pos, params).value
    M_B_H = M_Bondi_Hill_trans(pos, stokes_peb, params)
    M_3D_2DH = M_3D_2DH_trans(pos, stokes_peb, params)
    M_3D_2DB = M_3D_2DB_trans(pos, stokes_peb, params)

    # Plot all lines
    line_BH, = axs.loglog(pos.to(u.au).value, M_B_H.to(u.M_earth).value, color = "navy") #label="$M_{trans}$ Bondi to Hill"
    line_2DH_3D, = axs.loglog(pos.to(u.au).value, M_3D_2DH.to(u.M_earth).value, color="gold") # label="$M_{trans}$ 3D to 2DH accretion"
    line_2DB_3D, = axs.loglog(pos.to(u.au).value, M_3D_2DB.to(u.M_earth).value, color="green") #label="$M_{trans}$ 3D to 2DB accretion"

    # Find the mask where M_3D_2DH drops below M_B_H
    mask_below = M_3D_2DH < M_B_H
    # Find the mask where M_3D_2DB rises above M_B_H
    mask_above = M_3D_2DB > M_B_H

    # Plot dashed lines for invalid regions
    #line2_masked, = axs.loglog(pos.to(u.au)[mask_below], M_3D_2DH[mask_below].to(u.M_earth), color="gold", linestyle="dashed", alpha = 0.5)
    #line3_masked, = axs.loglog(pos.to(u.au)[mask_above], M_3D_2DB[mask_above].to(u.M_earth), color="green", linestyle="dashed", alpha = 0.5)

    # Plot masked lines
    line_2DH_3D.set_data(pos.to(u.au)[~mask_below].value, M_3D_2DH[~mask_below].to(u.M_earth).value)
    line_2DB_3D.set_data(pos.to(u.au)[~mask_above].value, M_3D_2DB[~mask_above].to(u.M_earth).value)

    line_iso = axs.loglog(pos.to(u.au).value, M_peb_iso(pos, params).to(u.M_earth).value, alpha = 0.5, color = "purple", linestyle ='--', label =" pebble isolation mass" )[0]
    #line_M_in = axs.loglog(sim_params.a_p0.to(u.au).value, sim_params.m0.to(u.M_earth).value, alpha = 0.3, linestyle ='--', label = "initial mass" )[0]

    stokes_peb = st_frag(pos, params).value
    M_B_H = M_Bondi_Hill_trans(pos, stokes_peb, params)
    M_3D_2DH = M_3D_2DH_trans(pos, stokes_peb, params)
    M_3D_2DB = M_3D_2DB_trans(pos, stokes_peb, params)

    
    mid_idx = len(pos) // 2
    off = 10
    label_line(line_iso, "Pebble isolation mass", pos.to(u.au)[mid_idx+11].value, M_peb_iso(pos, params).to(u.M_earth)[mid_idx+11].value+off, color="purple")

    offset = 10
    label_line(line_BH, "Hill", pos.to(u.au)[mid_idx-20].value, M_B_H.to(u.M_earth)[mid_idx-20].value+offset, color="navy")
    label_line(line_BH, "Bondi", pos.to(u.au)[mid_idx-20].value, M_B_H.to(u.M_earth)[mid_idx-20].value-19*offset, color="navy")

    offset2 = 50
    label_line(line_2DH_3D, "2D Hill", pos.to(u.au)[mid_idx-13].value, M_3D_2DH.to(u.M_earth)[mid_idx-5].value+offset2, rotation = -45, color="gold")
    label_line(line_2DH_3D, "3D", pos.to(u.au)[mid_idx-13].value, M_3D_2DH.to(u.M_earth)[mid_idx-5].value-offset2, rotation = -45, color="gold")

    #label_line(line_2DB_3D, "2D Bondi", pos.to(u.au)[mid_idx].value, M_3D_2DB.to(u.M_earth)[mid_idx].value, color="green")
    #label_line(line_2DB_3D, "3D", pos.to(u.au)[mid_idx].value, M_3D_2DB.to(u.M_earth)[mid_idx].value, color="green")

    mid_idx = len(sim_params.a_p0) // 2
    #label_line(line_M_in, "Intitial mass", sim_params.a_p0.to(u.au)[mid_idx].value, sim_params.m0.to(u.M_earth)[mid_idx].value, color="lightblue")
        

def line_text_handler(x_values, y_values):
    """Computes the midpoint of the line and the inclination to place the text on it"""

    # Calculate the midpoint and corner points
    mid_index = len(x_values) // 2
    start_x, mid_x, end_x = x_values[0], x_values[mid_index], x_values[-1]
    start_y ,mid_y, end_y = y_values[0], y_values[mid_index], y_values[-1]

    # Calculate the angle of the line
    angle = np.degrees(np.arctan2(end_y - start_y, end_x - start_x))

    return start_x, start_y, mid_x, mid_y, end_x, end_y, angle




def label_line(line, label, x, y, color='0.5', size=12, **kwarg):
    """Add a label to a line, at the proper angle.

    Arguments
    ---------
    line : matplotlib.lines.Line2D object,
    label : str
    x : float
        x-position to place center of text (in data coordinates)
    y : float
        y-position to place center of text (in data coordinates)
    color : str
    size : float
    """
    xdata, ydata = line.get_data()
    x1 = xdata[0]
    x2 = xdata[-1]
    y1 = ydata[0]
    y2 = ydata[-1]

    ax = line.axes
    text = ax.annotate(label, xy=(x, y), xytext=(-10, 0),
                       textcoords='offset points',
                       size=size, color=color,
                       horizontalalignment='left',
                       verticalalignment='bottom', **kwarg)
    sp1 = ax.transData.transform_point((x1, y1))
    sp2 = ax.transData.transform_point((x2, y2))

    rise = (sp2[1] - sp1[1])
    run = (sp2[0] - sp1[0])

    slope_degrees = np.degrees(np.arctan2(rise, run))
    text.set_rotation(slope_degrees)

    return text


def r_visc_irr(pos, t, params, sim_params):
    """computes the first transition from irradiated to viscous, for plotting purposes"""
    if params.H_r_model != 'irradiated':
        a = np.where(H_R_irr(pos, params) > H_R_visc_Lambrechts(pos, t, params))[0]
        print(a)
        if len(a)==0:
            return
        else:
            return pos[a[-1]]


def kepler11(axs, color):
    """Plots the kepler 11 system"""
    # Kepler 11 system
    a_p0 = np.array([0.091, 0.107, 0.155, 0.195, 0.250, 0.466])*u.au
    m0 = np.array([2.78, 5, 8.13, 9.48, 2.43, 25])*u.M_earth
    # Plot the planets
    axs.plot(a_p0, m0, "+", markersize=10, color = color,  zorder = 100)

def solar_system(axs, color):

    a_p0 = np.array([0.39, 0.72, 1, 1.52, 5.2, 9.54, 19.2, 30.06 ])*u.au
    m0 = np.array([0.055, 0.815, 1, 0.107, 317.8, 95.2, 14.5, 17.1])*u.M_earth
    axs.plot(a_p0, m0, "+", markersize=10, color = color, zorder = 100)

def HD191939 (axs, color):
    """System of planets TOI-1339, Orell-Miquell et al. 2022"""
	
    a_p0 = np.array([0.0804, 0.1752, 0.2132, 0.407, 0.812, 4.8 ])*u.au
    m0 = np.array([10.00, 8.0, 2.80, 112.2, 13.5, 660])*u.M_earth
    axs.plot(a_p0, m0, "+", markersize=10, color = color, zorder = 100)

def HD219134 (axs, color):
    """System of planets TOI-1469, Vogt et al. 2015"""

    a_p0 = np.array([3.11, 0.3753, 0.23508,	0.14574, 0.064816, 0.0384740])*u.au
    m0 = np.array([108, 11, 21, 8.9, 3.5, 3.8])*u.M_earth
    axs.plot(a_p0, m0, "+", markersize=10, color = color, zorder = 100)


def plot_roman_sensitivity(fig, ax, roman = True, kepler = True, solar_system = True, ss_moons = True, roman_sensitivity= False):
    """"Script to plot the roman sensitivity curves from https://github.com/mtpenny/wfirst-ml-figures/tree/master/sensitivity"""
    
    import json
    #Add the Solar System planet images
    #Uses the solution by Joe Kington from https://stackoverflow.com/questions/22566284/matplotlib-how-to-plot-images-instead-of-points
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox


    def imscatter(x, y, image, ax=None, zoom=1):
        if ax is None:
            ax = plt.gca()
        try:
            image = plt.imread(image)
        except TypeError:
            # Likely already an array...
            pass
        im = OffsetImage(image, zoom=zoom)
        x, y = np.atleast_1d(x, y)
        artists = []
        for x0, y0 in zip(x, y):
            ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
            artists.append(ax.add_artist(ab))
        ax.update_datalim(np.column_stack([x, y]))
        ax.autoscale()
        return artists
        

    if solar_system:
        planetsize=0.1
        imscatter([0.387098],[0.055], 'roman_sensitivity/mercury.png',ax=ax,zoom=planetsize*0.3)
        imscatter([0.723332],[0.815], 'roman_sensitivity/venus.png',ax=ax,zoom=planetsize*0.9)
        imscatter([1],[1],'roman_sensitivity/Earth_Western_Hemisphere_transparent_background.png',ax=ax,zoom=planetsize)
        imscatter([1.523679],[0.107], 'roman_sensitivity/mars.png',ax=ax,zoom=planetsize*240/500.0*0.9)
        imscatter([5.204267],[317.8],'roman_sensitivity/jupiter.png',ax=ax,zoom=planetsize)
        imscatter([9.582017],[95.152],'roman_sensitivity/saturn.png',ax=ax,zoom=planetsize*1.35)
        imscatter([19.229411],[15.91],'roman_sensitivity/uranus.png',ax=ax,zoom=planetsize)
        imscatter([30.103662],[17.147],'roman_sensitivity/neptune.png',ax=ax,zoom=planetsize)
    if ss_moons:
        planetsize=0.1
        imscatter([1],[0.7349/59.736],'roman_sensitivity/moon.png',ax=ax,zoom=planetsize*0.5)
        imscatter([5.204267],[1.4819/59.736],'roman_sensitivity/ganymede.png',ax=ax,zoom=planetsize)
        imscatter([9.582017],[1.346/59.736],'roman_sensitivity/titan.png',ax=ax,zoom=planetsize*0.22)


    #Axis limits
    Mmin=0.01
    Mmax=10000 #15000
    amin=0.009
    amax=100

    mjup=317.83
    msun=1/3.0024584e-6   


    amin=0.009
    amax=100

    if roman:
        ## ROMAN SENSITIVITY LINE
        nroaltparfile = 'roman_sensitivity/fitmaplimitsolns_NRO_layout_7f_3_covfac_52_3.json'
        nroaltpars = json.load(open(nroaltparfile,'r'))

        def nroalt(x,pars):
            return pars['a'] + pars['b']*x + pars['g']*np.sqrt(pars['d']**2+(x-pars['e'])**2)

        fittedx = np.arange(np.log10(amin)-1,np.log10(amax)+1,0.05)
        fittedline=nroalt(fittedx,nroaltpars)
        ax.plot(10**fittedx,10**fittedline,'-',color='b',lw=3)
        ax.text(20,0.17,'$Roman$',color='b',rotation=45)
    if roman_sensitivity:
        ### ROMAN SENSITIVITY CONTOURS
        smap = np.loadtxt('roman_sensitivity/all.magrid.NRO.layout_7f_3_covfac.52.filled') 
        x = 10**smap[:33,0]
        y = 10**smap[::33,1]
        print(smap.shape,x.shape[0]*y.shape[0])
        X,Y=np.meshgrid(x,y)
        z = smap[:,2].reshape(X.shape) 
        print("x",x)
        print("y",y)
        #Contours in log sensitivity
        cf = ax.contourf(X,Y,z,cmap='Blues',levels=[-1,-0.5,0,0.5,1,1.5,2,2.5,3,3.5,4,4.5],vmin=-1,vmax=8)
        cbar = plt.colorbar(cf,ax=ax,label='$Roman$ Sensitivity $-$ the number of planet detections\n expected if there is 1 planet per star at $(a,M_{\\rm p})$',ticks=[-1,0,1,2,3,4])
        cbar.ax.set_yticklabels(['0.1','1','10','100','1000','10000'])

    if kepler:
        ### KEPLER SENSITIVITY LINE
        #The Kepler line
        kepx = np.arange(np.log10(amin),np.log10(amax),0.05)
        def kep(x):
            ret = np.zeros(x.shape) + Mmax*100.1
            xx = 10**x
            xm = (xx<1.2)
            ret[xm] = 0.68*xx[xm]**0.75
            return ret 

        def kepburke2015(x):
            ret = np.zeros(x.shape) + Mmax*100.1
            xx = 10**x
            a0 = (530.0/365.25)**(2.0/3.0)
            xm = (xx<a0)
            ret[xm] = 2.2*(xx[xm]/a0)**0.75
            return ret 

        ax.plot(10**kepx,kepburke2015(kepx),'-',color='r',lw=3)
        ax.text(0.011,0.085,'$Kepler$',color='r',rotation=23)

    # ax.set_axisbelow(False)
    # ax.set_xlabel('$a_{\mathrm {p}}$ [AU]')
    # ax.set_ylabel('$M_{\mathrm {p}} [M_{\oplus}]$')
    # ax.set_xscale("log")
    # ax.set_yscale("log")
    # ax.set_xlim([amin,amax])
    # ax.set_ylim([Mmin,Mmax])
    # import matplotlib.ticker as ticker
    # ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    # ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    # ax.legend(loc=3,mode="expand",bbox_to_anchor=(0.0,0.995,1.0,0.102),ncol=3,fontsize=12,numpoints=1,handletextpad=-0.5)
    # plt.tight_layout()

    plt.savefig("figures/roman_sensitivity", dpi=300)
