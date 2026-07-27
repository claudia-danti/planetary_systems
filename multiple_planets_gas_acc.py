from dataclasses import dataclass
from typing import Optional, Union, Callable
import numpy as np
import astropy.units as u
import astropy.constants as const
from scipy.integrate import solve_ivp
from dataclasses import dataclass, field
import os
from functions import *
from functions_pebble_accretion import *
import h5py
from tqdm import tqdm


#### UNITS AND CONVERSIONS ####
Gauss_to_au_M_E_myr = (1*u.cm**(-1/2)*u.g**(1/2)/u.s).to(u.au**(-1/2)*u.M_earth**(1/2)/u.Myr).value
erg_cgs = (1*u.erg).to(u.cm**2*u.g/u.s**2).value
erg_s_to_au_M_E_Myr = (1*u.erg/u.s).to(u.au**2*u.M_earth/u.Myr**3).value

# routine to convert the dicts so that they can be stored in the hdf5 file, since hdf5 does not support mixed types
def save_dict_to_hdf5(grp, d):
    """Recursively save a dict to an hdf5 group, handling mixed types."""
    for k, v in d.items():
        key = str(k)
        if v is None:
            grp.attrs[key] = 'None'
        elif isinstance(v, str):
            grp.attrs[key] = v
        elif isinstance(v, (int, float, bool)):
            grp.attrs[key] = v
        elif isinstance(v, u.Quantity):
            ds = grp.create_dataset(key, data=np.atleast_1d(v.value))
            ds.attrs['unit'] = str(v.unit)
        elif isinstance(v, np.ndarray):
            if v.dtype == object:
                # object arrays: convert to strings
                grp.attrs[key] = str(v.tolist())
            else:
                grp.create_dataset(key, data=np.atleast_1d(v))
        elif isinstance(v, dict):
            sub_grp = grp.create_group(key)
            save_dict_to_hdf5(sub_grp, v)
        elif isinstance(v, (list, tuple)):
            try:
                arr = np.array(v)
                if arr.dtype == object:
                    grp.attrs[key] = str(v)  # fallback: save as string
                else:
                    grp.create_dataset(key, data=arr)
            except Exception:
                grp.attrs[key] = str(v)
        else:
            grp.attrs[key] = str(v)  # fallback for anything else

## routine to write on file hdf5 the simulation results and the parameters of the simulation
def save_simulation_hdf5(simulation, params, sim_params, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    fname = os.path.join(output_folder, f'simulation_{params.H_r_model}_Z_{params.Z}.h5')
    
    with h5py.File(fname, 'w') as f:
        sim_grp = f.create_group('simulation')
        for key, val in simulation.__dict__.items():
            if isinstance(val, u.Quantity):
                ds = sim_grp.create_dataset(key, data=np.atleast_1d(val.value))
                ds.attrs['unit'] = str(val.unit)
            elif isinstance(val, np.ndarray):
                if val.dtype == object:
                    sim_grp.attrs[key] = str(val.tolist())
                else:
                    sim_grp.create_dataset(key, data=val)
            elif isinstance(val, dict):
                grp = sim_grp.create_group(key)
                save_dict_to_hdf5(grp, val)
            elif val is None:
                sim_grp.attrs[key] = 'None'

        par_grp = f.create_group('params')
        for key, val in params.__dict__.items():
            if callable(val):
                continue
            if isinstance(val, (int, float, str, bool)):
                par_grp.attrs[key] = val
            elif val is None:
                par_grp.attrs[key] = 'None'

        sp_grp = f.create_group('sim_params')
        for key, val in sim_params.__dict__.items():
            if isinstance(val, np.ndarray):
                sp_grp.create_dataset(key, data=val)
            elif isinstance(val, (int, float, str, bool)):
                sp_grp.attrs[key] = val

@dataclass
class Params:
    #stellar parameters 
    star_mass: float = (const.M_sun).to(u.M_earth).value
    star_radius: float = field(init=False)  # Will be set in __post_init__  # float = (const.R_sun).to(u.au).value
    star_luminosity: float = field(init=False)  # Will be set in __post_init__  # (const.L_sun.cgs).value *erg_s_to_au_M_E_Myr
    star_magnetic_field: float = 1e3*Gauss_to_au_M_E_myr #=1kG
    M_dot_gas_star: Union[float, str] = "star_mass_linear" #Hartmann_2016, Liu_2019, star_mass_linear, star_mass_quadratic
    mdot_star_func: Callable[[float], float] = field(init=False, repr=False)

    #disc parameters
    iso_filtering: float = 1
    tau_disc: float= (5 * u.Myr).value
    disc_opacity: float = 1e-2
    Z: float = 0.01 
    alpha: float = 1e-2
    alpha_frag: float = 1e-4
    alpha_frag_out: float = 1e-4
    alpha_frag_in: float = 10*alpha_frag_out
    alpha_z: float = 1e-4
    alpha_z_out: float = 1e-4   #value of alpha_z outside the iceline if iceline_alpha_change (replaces alpha_z)
    alpha_z_in: float = 10*alpha_z_out  #value of alpha_z outside the iceline if iceline_alpha_change (replaces alpha_z)
    iceline_radius: Optional[float] = None
    St_const: Optional[float] = None
    v_frag: float = (1 * u.m/u.s).to(u.au/u.Myr).value
    v_frag_out: float = (10 * u.m/u.s).to(u.au/u.Myr).value
    v_frag_in: float = v_frag_out/10
    H_r_model: str = "Lambrechts_mixed"  #4 possible models: Ida_mixed, Liu_mixed, irradiated (flared Ida), flared (Bitsch), flat, Lambrechts_mixed
    dlnP_dlnR: float = -2
    epsilon_el: float = 1e-2
    epsilon_heat: float = 0.5
    a_gr: u.Quantity  = (0.1*u.mm).to(u.au).value
    rho_gr: u.Quantity  = (1*u.g/u.cm**3).to(u.M_earth/u.au**3).value
    mu: float = 2.34 #mean molecular weight
    cross_sec_H: u.Quantity = (2e-15*u.cm**2).value #collisional cross section of H2
    kappa: u.Quantity = (0.005*u.m**2/u.kg).to(u.au**2/u.M_earth).value #envelope opacity
    beta0: float = (500 * u.g / u.cm**2).to(u.earthMass / u.au**2).value #surface density of the envelope
    epsilon_p: float = 0.5
    epsilon_d: float = 0.05
    iceline_alpha_change: bool = False
    iceline_alpha_frag_change: bool = False
    iceline_v_frag_change: bool = False
    iceline_flux_change: bool = False
    resonance_trapping: bool = True
    gas_accretion: bool = True
    self_gravity: bool = False

    def __post_init__(self):
        # Set star_luminosity as a function of star_mass
        self.star_luminosity = L_star(self.star_mass)
        self.star_radius = R_star(self.star_mass)
        self.mdot_star_func = self._build_mdot_star_func()

    def update_alpha_z_iceline(self, pos, iceline_radius):
        if pos < iceline_radius:
            self.alpha_z = self.alpha_z_in
        else:
            self.alpha_z = self.alpha_z_out

    def update_alpha_frag_iceline(self, pos, iceline_radius):
        if pos < iceline_radius:
            self.alpha_frag = self.alpha_frag_in
        else:
            self.alpha_frag = self.alpha_frag_out

    def update_v_frag_iceline(self, pos, iceline_radius):
        if pos < iceline_radius:
            self.v_frag = self.v_frag_in
        else:
            self.v_frag = self.v_frag_out

    #to try to speed up the code and not have to check every time M_dot_star gets called        
    def _build_mdot_star_func(self):
            if self.M_dot_gas_star == "Liu_2019":
                return lambda t: M_dot_star_t_Mstar(t, self)
            if self.M_dot_gas_star == "Hartmann_2016":
                return M_dot_star_t
            if self.M_dot_gas_star == "star_mass_linear":
                return lambda t: M_dot_star_linear_scaling(t, self)
            if self.M_dot_gas_star == "star_mass_quadratic":
                return lambda t: M_dot_star_quadratic_scaling(t, self)
            return lambda t: self.M_dot_gas_star

@dataclass
class SimulationParams:
    # integration parameters
    t_in: float = 0.1 #Myr
    t_fin: float = 5 #Myr
    N_step: float = 10000
    
    tolerance: float = 1e-2
    mig_tolerance: float = 1e-3
    c: float = 1
    # sepcifying the step size needs to be computed based on the input of N_step
    step_size: float = field(init=False)
    t: np.array = field(init=False)
    ## new syntax cause lese Python complains
    m0: np.ndarray = field(default_factory=lambda: np.array([1e-3]*5))
    a_p0: np.ndarray = field(default_factory=lambda: np.array([20,15,10,5,2]))
    t0: np.ndarray = field(default_factory=lambda: 0.1 * np.ones(5))

    def __post_init__(self):
            self.step_size = (self.t_fin - self.t_in) / self.N_step
            self.t = np.geomspace(self.t_in, self.t_fin, int(self.N_step))

  
    @property
    def nr_planets(self):
        return len(self.a_p0)


@dataclass
class SimulationResults:
    time: u.Quantity
    mass: u.Quantity
    position: u.Quantity
    dM_dt: u.Quantity
    dR_dt: u.Quantity
    filter_fraction: u.Quantity
    flux_on_planet: u.Quantity
    F0: u.Quantity
    flux_ratio: u.Quantity
    sigma_peb: u.Quantity
    sigma_gas: u.Quantity
    acc_regimes: dict
    gas_acc_dict: dict


params = Params()
sim_params = SimulationParams()
peb_acc = PebbleAccretion()
gas_acc = GasAccretion()


def evolve_system(
    times, masses, positions, migration, filtering, peb_acc, gas_acc, params, sim_params):
    """Function that computes the dM/dt, dR/dt and filter fraction"""

    # checks if the planets are in the correct order, throws error if they swap position
    # if not np.all(positions[:-1] >= positions[1:]):
    # print('ERROR: PLANETS IN WRONG ORDER')
    # exit()
    # Allocate empty arrays of filled with 0's
    # They are matrixes: [number of planets x times]
    M_dot = np.zeros_like(masses) 
    R_dot = np.zeros_like(positions)
    sigma_peb = np.zeros_like(masses) 
    sigma_gas = np.zeros_like(masses) 

    # Flux on the planet i is obtained as: F_i = prod_0^i ( F0 * (1-f_i) )  with f_i filter fraction of the planet i
    flux_reduction = 1   # initial reduction of flux (1D vector of timestep -> is an intermediate quantity updated every timestep)
    filter_frac = np.zeros(masses.shape)  # accreted pebble fraction on the planets (2D matrix [planets x times])
    flux_on_planet = np.zeros_like(masses) # accreted pebble fraction on the planets (2D matrix [planets x times])
    flux_ratio = np.zeros(masses.shape) # ratio of the accreted pebble flux and the incoming flux
    
    # disc quantities related to time only
    mdot_star = params.mdot_star_func(times)    
    R_mag_cav = r_magnetic_cavity(mdot_star, params)
     ###### NOMINAL FLUX ########
    F0_nominal = flux_dtg_t(mdot_star, params)

    pos_previous = np.zeros_like(positions) #to check if the planets overtake each other
    pos_out = positions[0] #to kill the planets if they overtake each other
    
    for i in range(sim_params.nr_planets):

        # Iceline treatment: cuts the flux in half, increases the vertical stirring
        if params.iceline_alpha_change:
            params.update_alpha_z_iceline(positions[i], iceline(mdot_star, 170, params))
        if params.iceline_alpha_frag_change:
            params.update_alpha_frag_iceline(positions[i], iceline(mdot_star, 170, params))
        if params.iceline_v_frag_change:
            params.update_v_frag_iceline(positions[i], iceline(mdot_star, 170, params))
             
        if params.iceline_flux_change:
            if params.iceline_radius == None:
                iceline_radius = iceline(mdot_star, 170, params)
            else:
                iceline_radius = params.iceline_radius
            F0 = np.where(positions[i] < iceline_radius, 1/2, 1)*F0_nominal
        else: 
            F0 = F0_nominal

        #disc quantities related to planet position 
        H_r = H_R(positions[i], mdot_star, params)
        Sigma_gas = sigma_gas_steady_state(positions[i], H_r, mdot_star, params)

       # to flag the accretion regime we are in
        peb_acc._set_planet_id (i)
        peb_acc.create_dict_planet_entry(i)
        # to flag the gas accretion regime we are in
        gas_acc._set_planet_id (i)
        gas_acc.create_dict_planet_entry(i)
        #Diff equation for mass growth [planets x times]
        M_dot[i], sigma_peb[i], sigma_gas[i], acc_regimes = peb_acc.dMc_dt_f(times, masses[i], positions[i], mdot_star, H_r, Sigma_gas, F0, flux_reduction, params)

        # delaying the embryo
        time_mask = times < sim_params.t0[i]
        M_dot[i, time_mask] = 0  # the delayed embryo does not grow

        # option for migration
        if migration:
                
            R_dot[i] = dR_dt_both(times, positions[i], masses[i], H_r, Sigma_gas, params) #includes type II prescription

            H_r_previous = H_R(positions[i-1], mdot_star, params) #needed to compute the resonance condition
            # I want the resonance trapping to be activated only after the planets reach pebble isolation mass
            if params.resonance_trapping and masses[i] > M_peb_iso(H_r, params) and masses[i-1] > M_peb_iso(H_r_previous, params):
                if ((pos_out/positions[i])**(3/2))<2 and ((pos_out/positions[i])**(3/2))!= 1:
                    # outer planet gets trapped in resonance
                    R_dot[i-1] = 0
                    positions[i-1] = meanmr_two_one_out(positions[i])

                else:
                    # planet reaches inner edge
                    dead_by_mig = (positions[i] < R_mag_cav) #inward drifting magentic cavity radius
                    R_dot[i, dead_by_mig] = 0  # dR/dt = 0 in case the planet has reached the inner edge 
                    positions[i, dead_by_mig] = R_mag_cav  # set the position to the inner edge
                                
                    # if dead_by_mig:
                    #     print("Planet "+str(positions[i])[:4]+" reached the inner edge")
                    #     print("R_planet", positions[i])
                    #     print("magentic cavity", R_mag_cav)
            else:
                #regardless of iso, check if the planet has reached the inner edge
                dead_by_mig = (positions[i] < R_mag_cav)
                R_dot[i, dead_by_mig] = 0  # dR/dt = 0 in case the planet has reached the inner edge 
                M_dot[i, dead_by_mig] = 0  # dM/dt = 0 in case the planet has reached the inner edge
                positions[i, dead_by_mig] = R_mag_cav  # set the position to the inner edge
            
                # if dead_by_mig:
                #     print("Planet "+str(positions[i])[:4]+" reached the inner edge")
                #     print("R_planet", positions[i])
                #     print("magentic cavity", R_mag_cav)
                
            #to check if the planets overtake each other
            for j in range(i):
                if positions[i]>pos_previous[j]:
                    # I kill the inner planet in the collision
                    R_dot[i] = 0
                    M_dot[i] = 0

            #the temporary position is the outer planet
            pos_out = positions[i]
            pos_previous = positions.copy()
        if masses[i] > M_peb_iso(H_r, params):
            # to give the option of not having gas accretion (for the timescale plots)
            if params.gas_accretion:
                M_dot[i], gas_accretion_dict = gas_acc.dMc_dt_gas(times, masses[i], positions[i], Sigma_gas, H_r, params)
            else:
                M_dot[i] = 0
        else:
            gas_accretion_dict = None #just otherwise the dict is not defined

        flux_on_planet[i] = F0 * flux_reduction
        filter_frac[i] = np.clip(M_dot[i] / flux_on_planet[i], 0, 1)  # filtering fraction due to the planet i is restricted between [0,1]
        filter_frac[i, flux_on_planet[i] == 0] = 0  # when one planet reaches peb iso the definition of ff is 0/0, this prevents the code from crushing
        dead = (masses[i] > M_peb_iso(H_r, params)) | (positions[i] < R_mag_cav)  # cut the simulation once it reaches pebble isolaton mass or inner edge
        #filter_frac[i, dead] = 1 # in case the planet reaches peb iso or inner cutoff the ff is 1
        filter_frac[i, dead] = params.iso_filtering # in case the planet reaches peb iso or inner cutoff the ff is 1
        flux_ratio[i] = flux_on_planet[i]/F0

        if filtering:
            flux_reduction *= (1 - filter_frac[i])  # amount that is multiplied by F0 to get F_i
        
    if np.any(np.isnan(filter_frac)):
        print("Nan in ff")
    #return M_dot, R_dot, filter_frac, flux_on_planet, F0, flux_ratio, R_acc, H_peb, R_acc_H, R_acc_B, M_dot_twoD_B, M_dot_twoD_H, M_dot_threeD_B, M_dot_threeD_H,  M_dot_threeD_unif, sigma_peb, sigma_gas, H_r, acc_regimes, gas_accretion_dict
    return M_dot, R_dot, filter_frac, flux_on_planet, F0, flux_ratio, sigma_peb, sigma_gas, acc_regimes, gas_accretion_dict


def simulate_euler(migration, filtering, peb_acc, gas_acc, params, sim_params, output_folder='sims/gas_acc'):
    """Euler solver for the differential equation"""
    args = (migration, filtering, peb_acc, gas_acc, params, sim_params)
    # Initialize lists to store values
    t_values = [sim_params.t[0]]
    mass_values = [sim_params.m0]
    pos_values = [sim_params.a_p0]

    # Run the first time the diff eq to have the right first values for the other quantities
    m_dot, r_dot, filter_f, flux_p, flux, flux_ratio, Sigma_peb, Sigma_gas, acc_regimes, gas_acc_dict = evolve_system(t_values[0], mass_values[0], pos_values[0], *args)

    Mdot_values = [m_dot]
    Rdot_values = [r_dot]
    filter_values = [filter_f]
    planet_flux_values = [flux_p]
    F0_values = [flux]
    flux_ratio_values = [flux_ratio]
    sigma_peb = [Sigma_peb]
    sigma_gas = [Sigma_gas]

    c = 10000
    total_steps = int((sim_params.t_fin - sim_params.t_in) / sim_params.step_size)

    with tqdm(total=total_steps, desc="Simulating", unit="step") as pbar:

        while t_values[-1] < sim_params.t_fin:
            # Rename the end of each vector
            t = t_values[-1]
            m0 = mass_values[-1]
            p0 = pos_values[-1]

            # Euler integrator
            #mdot, rdot, ff, F_p, F0, F_ratio, racc, hpeb, raccH, raccB, mdottwoDB, mdottwoDH, mdotthreeDB, mdotthreeDH, mdotthreeDunif, sigmapeb, sigmagas, Hr, acc_regimes, gas_acc_dict = evolve_system(t, m0, p0, *args)
            mdot, rdot, ff, F_p, F0, F_ratio, sigmapeb, sigmagas, acc_regimes, gas_acc_dict = evolve_system(t, m0, p0, *args)

            m = sim_params.step_size * mdot + m0
            p = sim_params.step_size * rdot + p0
            # Append values to lists
            t_values.append(t + sim_params.step_size)

            mass_values.append(m)
            pos_values.append(p)
            Mdot_values.append(mdot)
            Rdot_values.append(rdot)
            filter_values.append(ff)
            planet_flux_values.append(F_p)
            F0_values.append(F0)
            flux_ratio_values.append(F_ratio)
            sigma_peb.append(sigmapeb)
            sigma_gas.append(sigmagas)

            pbar.update(1)
            # Show current sim time in the bar
            pbar.set_postfix({"t [Myr]": f"{t:.3f}"})

    # Convert lists to arrays
    t_values = np.array(t_values) * u.Myr
    mass_values = np.array(mass_values).T * u.M_earth
    pos_values = np.array(pos_values).T * u.au
    Mdot_values = np.array(Mdot_values).T * u.M_earth / u.Myr
    Rdot_values = np.array(Rdot_values).T * u.au / u.Myr
    filter_values = np.array(filter_values).T
    planet_flux_values = np.array(planet_flux_values).T * u.M_earth / u.Myr
    F0_values = np.array(F0_values).T * u.M_earth / u.Myr
    flux_ratio_values = np.array(flux_ratio_values).T
    sigma_peb = np.array(sigma_peb).T * u.M_earth / u.au**2
    sigma_gas = np.array(sigma_gas).T * u.M_earth / u.au**2
   
   # Create the SimulationResults object
    simulation = SimulationResults(t_values, mass_values, pos_values, Mdot_values, Rdot_values, filter_values, 
                                planet_flux_values, F0_values, flux_ratio_values, sigma_peb, sigma_gas, acc_regimes, gas_acc_dict)
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # # Construct file paths
    # sim_filename = os.path.join(output_folder, 'simulation_'+str(params.H_r_model)+'_e_el_'+str(params.epsilon_el)+'_vfrag_'+str(((params.v_frag*u.au/u.Myr).to(u.m/u.s)).value)+'_planets_'+str(sim_params.nr_planets)+'_t0_'+str(sim_params.t0[-1])+'_N_steps'+str(sim_params.N_step)+'_Mstar_'+str((params.star_mass*u.M_earth).to(u.M_sun).value)+'_Z_'+str(params.Z)+'.json')
    # sim_params_filename = os.path.join(output_folder, 'sim_params_'+str(params.H_r_model)+'_e_el_'+str(params.epsilon_el)+'_vfrag_'+str(((params.v_frag*u.au/u.Myr).to(u.m/u.s)).value)+'_planets_'+str(sim_params.nr_planets)+'_t0_'+str(sim_params.t0[-1])+'_N_steps'+str(sim_params.N_step)+'_Mstar_'+str((params.star_mass*u.M_earth).to(u.M_sun).value)+'_Z_'+str(params.Z)+'.json')
    # params_filename = os.path.join(output_folder, 'params_'+str(params.H_r_model)+'_e_el_'+str(params.epsilon_el)+'_vfrag_'+str(((params.v_frag*u.au/u.Myr).to(u.m/u.s)).value)+'_planets_'+str(sim_params.nr_planets)+'_t0_'+str(sim_params.t0[-1])+'_N_steps'+str(sim_params.N_step)+'_Mstar_'+str((params.star_mass*u.M_earth).to(u.M_sun).value)+'_Z_'+str(params.Z)+'.json')

    # Write the result to hdf5 files
    save_simulation_hdf5(simulation, params, sim_params, output_folder)    
    return simulation
