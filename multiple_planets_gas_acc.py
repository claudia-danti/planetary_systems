from dataclasses import dataclass
from typing import Optional

import numpy as np
import astropy.units as u
import astropy.constants as const
import json
from scipy.integrate import solve_ivp
from dataclasses import dataclass, field

from functions import *
from functions_pebble_accretion import *
from functions_plotting import *


class SimulationEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, u.Quantity):
            if isinstance(obj, np.ndarray):
                return obj.value.tolist()  # Convert ndarray to list
            return {'magnitude': obj.value, 'unit': str(obj.unit)}
        elif isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert ndarray to list
        elif isinstance(obj, list):  # Check if it's a list
            if isinstance(obj, u.Quantity):
                return [self.default(item.value) for item in obj] 
            return [self.default(item) for item in obj]  # Recursively encode each item
        return super().default(obj)


@dataclass
class Params:
    #stellar parameters 
    star_mass: u.Quantity = const.M_sun.cgs
    star_radius: u.Quantity = const.R_sun.cgs
    star_luminosity: u.Quantity = const.L_sun.cgs 
    star_magnetic_field: u.Quantity = 1e3*u.cm**(-1/2)*u.g**(1/2)/u.s #=1*u.G 1
    M_dot_star: Optional[u.Quantity] = None #1e-8*const.M_sun.cgs/((1*u.yr).to(u.s))

    #disc parameters
    iso_filtering: float = 1
    tau_disc: u.Quantity = (3 * u.Myr).to(u.s)
    disc_opacity: float = 1e-2
    Z: float = 0.01 
    alpha: float = 1e-3
    alpha_frag: float = 1e-4
    alpha_z: float = 1e-4
    alpha_z_out: float = 1e-4   #value of alpha_z outside the iceline if iceline_alpha_change (replaces alpha_z)
    alpha_z_in: float = 10*alpha_z_out  #value of alpha_z outside the iceline if iceline_alpha_change (replaces alpha_z)
    iceline_radius: Optional[u.Quantity] = (2 * u.au).to(u.cm)
    St_const: Optional[float] = None
    v_frag: u.Quantity = (1 * u.m/u.s).to(u.cm/u.s)
    H_r_model: str = "Lambrechts_mixed"  #4 possible models: Ida_mixed, Liu_mixed, irradiated (flared Ida), flared (Bitsch), flat, Lambrechts_mixed
    dlnP_dlnR: float = -2
    epsilon_el: float = 1e-2
    epsilon_heat: float = 0.5
    a_gr: u.Quantity  = (0.1*u.mm).to(u.cm)
    rho_gr: u.Quantity  = 1*u.g/u.cm**3
    mu: float = 2.34 #mean molecular weight
    cross_sec_H: u.Quantity = 2e-15*u.cm**2 #collisional cross section of H2
    kappa: u.Quantity = (0.005*u.m**2/u.kg).to(u.cm**2/u.g) #envelope opacity
    beta0: u.Quantity = 500 * u.g / u.cm**2
    epsilon_p: float = 0.5
    epsilon_d: float = 0.05
    iceline_alpha_change: bool = False
    iceline_flux_change: bool = True
    resonance_trapping: bool = True

    def update_alpha_z_iceline(self, pos, iceline_radius):
        if pos < iceline_radius:
            self.alpha_z = self.alpha_z_in
        else:
            self.alpha_z = self.alpha_z_out

@dataclass
class SimulationParams:
    # integration parameters
    t_in: u.Quantity = (1e5 * u.yr).to(u.s)
    t_fin: u.Quantity = (10 * u.Myr).to(u.s)
    N_step: float = 1
    tolerance: float = 1e-2
    mig_tolerance: float = 1e-3

    # sepcifying the step size needs to be computed based on the input of N_step
    step_size: float = field(init=False)
    t: np.array = field(init=False)
    t_span: tuple = field(init=False)

    # initial mass and position arrays for the two planets
    m0: u.Quantity = (np.array([(1e-3 * u.earthMass).to(u.g).value,
                                (1e-3 * u.earthMass).to(u.g).value,
                                (1e-3 * u.earthMass).to(u.g).value,
                                (1e-3 * u.earthMass).to(u.g).value,
                                (1e-3 * u.earthMass).to(u.g).value,
                                ])* u.g)
    
    a_p0: u.Quantity = (np.array([(20 * u.au).to(u.cm).value,
                                  (15 * u.au).to(u.cm).value,
                                  (10 * u.au).to(u.cm).value,
                                  (5 * u.au).to(u.cm).value,
                                  (1 * u.au).to(u.cm).value,
                                  ])* u.cm)

    t0: u.Quantity = ([1e5] * np.ones(len(a_p0)) * u.yr).to(u.s) # warning, this also goes in the initial conditions when doing mulitple planets otherwise it won't work
    
    def __post_init__(self):
            self.step_size = (self.t_fin - self.t_in) / self.N_step
            self.t = np.geomspace(self.t_in.value, self.t_fin.value, int(self.N_step)) * self.t_in.unit
            self.t_span = (self.t_in.value, self.t_fin.value)

  
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
    R_acc: u.Quantity
    H_peb: u.Quantity
    R_acc_H: u.Quantity
    R_acc_B: u.Quantity
    Mdot_twoD_B: u.Quantity
    Mdot_twoD_H: u.Quantity
    Mdot_threeD_B: u.Quantity
    Mdot_threeD_H: u.Quantity
    Mdot_threeD_unif: u.Quantity
    sigma_peb: u.Quantity
    sigma_gas: u.Quantity
    H_r : float
    acc_regimes: dict
    gas_acc_dict: dict


params = Params()
sim_params = SimulationParams()
peb_acc = PebbleAccretion()
gas_acc = GasAccretion()


def evolve_system(
    times: u.s, masses: u.g, positions: u.cm, migration, filtering, peb_acc, gas_acc, params, sim_params):
    """Function that computes the dM/dt, dR/dt and filter fraction"""

    # checks if the planets are in the correct order, throws error if they swap position
    # if not np.all(positions[:-1] >= positions[1:]):
    # print('ERROR: PLANETS IN WRONG ORDER')
    # exit()
    # Allocate empty arrays of the right units filled with 0's
    # They are matrixes: [number of planets x times]
    M_dot = np.zeros_like(masses) / u.s
    R_dot = np.zeros_like(positions) / u.s
    R_acc = np.zeros_like(positions)
    H_peb = np.zeros_like(positions)

    #debug quantities
    R_acc_H = np.zeros_like(positions)
    R_acc_B = np.zeros_like(positions)
    M_dot_twoD_B = np.zeros_like(masses) / u.s
    M_dot_twoD_H = np.zeros_like(masses) / u.s
    M_dot_threeD_B = np.zeros_like(masses) / u.s
    M_dot_threeD_H = np.zeros_like(masses) / u.s
    M_dot_threeD_unif = np.zeros_like(masses) / u.s

    sigma_peb = np.zeros_like(masses) / u.cm**2
    sigma_gas = np.zeros_like(masses) / u.cm**2
    H_r = np.zeros(positions.shape) #is a value without units

    # Flux on the planet i is obtained as: F_i = prod_0^i ( F0 * (1-f_i) )  with f_i filter fraction of the planet i
    flux_reduction = np.ones(1)   # initial reduction of flux (1D vector of timestep -> is an intermediate quantity updated every timestep)
    filter_frac = np.zeros(masses.shape)  # accreted pebble fraction on the planets (2D matrix [planets x times])
    flux_on_planet = (np.zeros_like(masses) / u.s)  # accreted pebble fraction on the planets (2D matrix [planets x times])
    flux_ratio = np.zeros(masses.shape) # ratio of the accreted pebble flux and the incoming flux
    
    #F0_nominal = (100*u.earthMass/u.Myr).to(u.g/u.s)
    ###### NOMINAL FLUX ########

    F0_nominal = flux_dtg_t(times, params)
    #F0_nominal = flux_const(times, params, sim_params)

    print("F0_nominal: ",F0_nominal.to(u.earthMass/u.Myr))

    pos_previous = np.zeros_like(positions)*u.cm #to check if the planets overtake each other
    pos_out = positions[0] #to kill the planets if they overtake each other
    
    for i in range(sim_params.nr_planets):

        # Iceline treatment: cuts the flux in half, increases the vertical stirring
        if params.iceline_alpha_change:
            params.update_alpha_z_iceline(positions[i], iceline(times, 170*u.K, params))
            
        if params.iceline_flux_change:
            if params.iceline_radius == None:
                iceline_radius = iceline(times, 170*u.K, params)
                print("current iceline position: ", iceline_radius.to(u.au))
            else:
                iceline_radius = params.iceline_radius
            F0 = np.where(positions[i] < iceline_radius, 1/2, 1)*F0_nominal
        else: 
            F0 = F0_nominal

        print("Planet "+str(positions[i].to(u.au))[:4]+", F0: ", F0.to(u.earthMass/u.Myr))
        print("Planet "+str(positions[i].to(u.au))[:4]+", M_dot_star: ", M_dot_star(times, params).to(u.M_sun/u.yr))

        # to flag the accretion regime we are in
        peb_acc._set_planet_id (i)
        peb_acc.create_dict_planet_entry(i)
        # to flag the gas accretion regime we are in
        gas_acc._set_planet_id (i)
        gas_acc.create_dict_planet_entry(i)


        # Diff equation for mass growth [planets x times]
        M_dot[i], R_acc[i], H_peb[i], R_acc_H[i], R_acc_B[i], M_dot_twoD_B[i], M_dot_twoD_H[i],M_dot_threeD_B[i], M_dot_threeD_H[i], M_dot_threeD_unif[i], sigma_peb[i], sigma_gas[i], H_r[i], acc_regimes = peb_acc.dMc_dt_f(times, masses[i], positions[i], F0, flux_reduction, params, sim_params)

        # delaying the embryo
        time_mask = times < sim_params.t0[i]
        M_dot[i, time_mask] = 0  # the delayed embryo does not grow

        # option for migration
        if migration:
            if params.H_r_model != "flat":
                sigma_gas_val = sigma_gas_steady_state(positions[i], times, params)
                print("Sigma gas:", sigma_gas_val)
            else:
                sigma_gas_val = sigma_gas_t(positions[i], times, params)
            
                
            #R_dot[i] = dR_dt(times, positions[i], masses[i], sigma_gas_val, params)
            R_dot[i] = dR_dt_both(times, positions[i], masses[i], sigma_gas_val, params) #includes type II prescription

            print("ratio of the positions", (pos_out/positions[i])**(3/2))

            # I want the resonance trapping to be activated only after the planets reach pebble isolation mass
            if params.resonance_trapping and masses[i] > M_peb_iso(positions[i], times, params) and masses[i-1] > M_peb_iso(positions[i-1], times, params):
                if ((pos_out/positions[i])**(3/2))<2 and ((pos_out/positions[i])**(3/2))!= 1:
                #if np.isclose((pos_out/positions[i])**(3/2), 2, rtol=1e-2, atol=1e-02, equal_nan=False) and ((pos_out/positions[i])**(3/2))!= 1:
                    #R_dot[i] = 0
                    R_dot[i-1] = 0
                    print("2:1 MMR reached for planets "+str(pos_out.to(u.au))[:4]+" and "+str(positions[i].to(u.au))[:4])
                    #positions[i] = meanmr_two_one_in(pos_out)
                    positions[i-1] = meanmr_two_one_out(positions[i])
                    print("New positions: ", positions[i].to(u.au), positions[i-1].to(u.au))
                    print("MMR ratio of positions ", (positions[i-1]/positions[i])**(3/2))
                else:
                    #to keep migrating the planets after they reach peb iso
                    dead_by_mig = (positions[i] < r_magnetic_cavity_gen(times, params))
                    R_dot[i, dead_by_mig] = 0  # dR/dt = 0 in case the planet has reached the inner edge 
                    positions[i, dead_by_mig] = r_magnetic_cavity_gen(times, params)  # set the position to the inner edge
                                
                    if dead_by_mig:
                        print("Planet "+str(positions[i].to(u.au))[:4]+" reached the inner edge")
                        print("R_planet", positions[i].to(u.au))
                        print("magentic cavity", r_magnetic_cavity_gen(times, params).to(u.au))
            else:
                #to keep migrating the planets after they reach peb iso
                dead_by_mig = (positions[i] < r_magnetic_cavity_gen(times, params))
                R_dot[i, dead_by_mig] = 0  # dR/dt = 0 in case the planet has reached the inner edge 
                M_dot[i, dead_by_mig] = 0  # dM/dt = 0 in case the planet has reached the inner edge
                positions[i, dead_by_mig] = r_magnetic_cavity_gen(times, params)  # set the position to the inner edge
                            
                if dead_by_mig:
                    print("Planet "+str(positions[i].to(u.au))[:4]+" reached the inner edge")
                    print("R_planet", positions[i].to(u.au))
                    print("magentic cavity", r_magnetic_cavity_gen(times, params).to(u.au))
                
            #to check if the planets overtake each other
            for j in range(i):
                if positions[i]>pos_previous[j]:
                    # I kill the inner planet in the collision
                    R_dot[i] = 0
                    M_dot[i] = 0
                    print("Planets overtook each other")

            #the temporary position is the outer planet
            pos_out = positions[i]
            pos_previous = positions.copy()


        if masses[i] > M_peb_iso(positions[i], times, params):
            M_dot[i], gas_accretion_dict = gas_acc.dMc_dt_gas(times, masses[i], positions[i], params, sim_params)
        else:
            #this is just otherwise the gas acc dict does not work
            gas_accretion_dict = None

        # The planets should stop at Jupiter mass
        dead_by_mass = masses[i] > (1*const.M_jup).to(u.g)
        M_dot[i, dead_by_mass] = 0  
        R_dot[i, dead_by_mass] = 0

        flux_on_planet[i] = F0 * flux_reduction
        filter_frac[i] = np.clip(M_dot[i] / flux_on_planet[i], 0, 1)  # filtering fraction due to the planet i is restricted between [0,1]
        filter_frac[i, flux_on_planet[i] == 0] = 0  # when one planet reaches peb iso the definition of ff is 0/0, this prevents the code from crushing
        dead = (masses[i] > M_peb_iso(positions[i], times, params)) | (positions[i] < r_magnetic_cavity_gen(times, params))  # cut the simulation once it reaches pebble isolaton mass or inner edge
        #filter_frac[i, dead] = 1 # in case the planet reaches peb iso or inner cutoff the ff is 1
        filter_frac[i, dead] = params.iso_filtering # in case the planet reaches peb iso or inner cutoff the ff is 1
        print("params iso filtering", params.iso_filtering)
        print("filter fraction after iso", filter_frac[i, dead])
        flux_ratio[i] = flux_on_planet[i]/F0

        if filtering:
            flux_reduction *= (1 - filter_frac[i])  # amount that is multiplied by F0 to get F_i
        
        print("dust to gas ratio, planet "+str(positions[i].to(u.au))[:4], sigma_peb[i]/sigma_gas[i])

    if np.any(np.isnan(filter_frac)):
        print("Nan in ff")
    return M_dot, R_dot, filter_frac, flux_on_planet, F0, flux_ratio, R_acc, H_peb, R_acc_H, R_acc_B, M_dot_twoD_B, M_dot_twoD_H, M_dot_threeD_B, M_dot_threeD_H,  M_dot_threeD_unif, sigma_peb, sigma_gas, H_r, acc_regimes, gas_accretion_dict


def simulate_euler(migration, filtering, peb_acc, gas_acc, params, sim_params):
    """Euler solver for the differential equation"""
    args = (migration, filtering, peb_acc, gas_acc, params, sim_params)

    # creating the arrays that will be updated every timestep by appending the solution, so far they are 1D arrays
    t_values = np.array([sim_params.t[0].value])*u.s # 1D with the t initial value
    mass_values = np.array([sim_params.m0.value])*u.g # 1D with the initial mass value
    pos_values = np.array([sim_params.a_p0.value])*u.cm # 1D with the initial position value
    
    # I run the first time the diff eq to have the right first values for the other quantities
    m_dot, r_dot, filter_f, flux_p, flux , flux_ratio, R_acc, H_peb, R_acc_H, R_acc_B, Mdot_twoD_B, Mdot_twoD_H, Mdot_threeD_B, Mdot_threeD_H, Mdot_threeD_unif, Sigma_peb, Sigma_gas, HR, acc_regimes, gas_acc_dict = evolve_system(t_values[0], mass_values[0], pos_values[0], *args)
    
    Mdot_values = np.array([m_dot])*u.g/u.s # 1D (nr_planets) with values of 0 
    Rdot_values = np.array([r_dot])*u.cm/u.s # 1D (nr_planets) with values of 0 
    filter_values = np.array([filter_f]) # 1D (nr_planets) with values of 0 
    planet_flux_values = np.array([flux_p])*u.g/u.s # 1D (nr_planets) with values of 0 
    F0_values = np.array([flux.value])*u.g/u.s # 1D (nr_planets) with values of flux (t, pos[0])
    flux_ratio_values = np.array([flux_ratio]) # 1D (nr_planets) with values of flux (t, pos[0])

    # debug quantities
    r_acc = np.array([R_acc])*u.cm
    h_peb = np.array([H_peb])*u.cm
    r_acc_h = np.array([R_acc_H])*u.cm
    r_acc_b = np.array([R_acc_B])*u.cm
    mdot_twod_bondi = np.array([Mdot_twoD_B])*u.g/u.s 
    mdot_twod_hill = np.array([Mdot_twoD_H])*u.g/u.s 
    mdot_threed_bondi = np.array([Mdot_threeD_B])*u.g/u.s 
    mdot_threed_hill = np.array([Mdot_threeD_H])*u.g/u.s 
    mdot_threed_unif = np.array([Mdot_threeD_unif])*u.g/u.s 
    sigma_peb = np.array([Sigma_peb])*u.g/u.cm**2
    sigma_gas = np.array([Sigma_gas])*u.g/u.cm**2
    H_r = np.array([HR])


    
    while t_values[-1] < sim_params.t_fin:

        # I rename the end of each vector
        t = t_values[-1]
        m0 = mass_values[-1]
        p0 = pos_values[-1]

        # Euler integrator
        mdot, rdot, ff, F_p, F0, F_ratio, racc, hpeb, raccH, raccB, mdottwoDB, mdottwoDH, mdotthreeDB, mdotthreeDH, mdotthreeDunif, sigmapeb, sigmagas, Hr, acc_regimes, gas_acc_dict = evolve_system(t, m0, p0, *args) 
        
        m = sim_params.step_size*(mdot) + m0
        p = sim_params.step_size*(rdot) + p0
        
        print("alpha in t loop", params.alpha)
        #only update the values if the error is smaller than the tolerance
        # addition of the computed approximations to get the next value of the function
        #t_values = np.append(t_values, (t + sim_params.step_size))
        t_values = np.append(t_values, (t + sim_params.step_size))
        mass_values = np.append(mass_values, [m], axis = 0)
        pos_values = np.append(pos_values, [p], axis = 0)
        Mdot_values = np.append(Mdot_values, [mdot], axis = 0)
        Rdot_values = np.append(Rdot_values, [rdot], axis = 0)
        filter_values = np.append(filter_values, [ff], axis = 0)
        planet_flux_values = np.append(planet_flux_values, [F_p], axis = 0)
        F0_values = np.append(F0_values, [F0], axis = 0)
        flux_ratio_values = np.append(flux_ratio_values, [F_ratio], axis = 0)

        #debug quantities
        r_acc = np.append(r_acc, [racc], axis = 0)
        h_peb = np.append(h_peb, [hpeb], axis = 0)
        r_acc_h = np.append(r_acc_h, [raccH], axis =0)
        r_acc_b = np.append(r_acc_b, [raccB], axis =0)
        mdot_twod_bondi = np.append(mdot_twod_bondi, [mdottwoDB], axis =0)
        mdot_twod_hill = np.append(mdot_twod_hill, [mdottwoDH], axis =0)
        mdot_threed_bondi = np.append(mdot_threed_bondi, [mdotthreeDB], axis =0)
        mdot_threed_hill = np.append(mdot_threed_hill, [mdotthreeDH], axis =0)
        mdot_threed_unif = np.append(mdot_threed_unif, [mdotthreeDunif], axis =0)
        sigma_peb = np.append(sigma_peb, [sigmapeb], axis=0)
        sigma_gas = np.append(sigma_gas, [sigmagas], axis=0)
        H_r = np.append(H_r, [Hr], axis=0)

    simulation = SimulationResults(t_values, mass_values.T, pos_values.T, Mdot_values.T, Rdot_values.T, filter_values.T, 
                                   planet_flux_values.T, F0_values.T, flux_ratio_values.T, r_acc.T, h_peb.T, r_acc_h.T, 
                                   r_acc_b.T, mdot_twod_bondi.T,mdot_twod_hill.T, mdot_threed_bondi.T, mdot_threed_hill.T, 
                                   mdot_threed_unif.T, sigma_peb.T, sigma_gas.T, H_r.T, acc_regimes, gas_acc_dict)
    # Write the result to a JSON file
    with open('sims/gas_acc/sim_'+str(sim_params.nr_planets)+'planets_'+str(params.H_r_model)+'_'+str(params.epsilon_el)+'e_el_'+str((params.v_frag).to(u.m/u.s).value)+'_t'+str(sim_params.N_step)+'.json', 'w') as file:
        json.dump(simulation.__dict__, file, cls=SimulationEncoder)

    return simulation

"""
    # Initial timestep
    dt = sim_params.step_size

    # Error tolerance
    tolerance = sim_params.tolerance
    migration_tolerance = sim_params.mig_tolerance

    
    while t_values[-1] < sim_params.t_fin:

        # I rename the end of each vector
        t = t_values[-1]
        m0 = mass_values[-1]
        p0 = pos_values[-1]

        # Euler integrator
        mdot, rdot, ff, F_p, F0, F_ratio, racc, hpeb, raccH, raccB, mdottwoDB, mdottwoDH, mdotthreeDB, mdotthreeDH, mdotthreeDunif, sigmapeb, sigmagas, acc_regimes = evolve_system(t, m0, p0, *args) 
        
        m = sim_params.step_size*(mdot) + m0
        p = sim_params.step_size*(rdot) + p0
        
        m = dt * mdot + m0
        p = dt * rdot + p0
        # Calculate error
        error = calculate_error(m0, m, p0, p)
        print("error",error)
        # Determine the appropriate tolerance
        current_tolerance = tolerance
        if np.abs(mdot.value).max() < 1e-8:  # Threshold for negligible mass accretion
            current_tolerance = migration_tolerance
            print("negligible mass accretion")
        # Adjust timestep based on error
        if error > current_tolerance:
            dt = dt / 5  # Reduce timestep
        else:
            if error < current_tolerance / 2:
                dt = dt * 1.5  # Increase timestep
        
            #only update the values if the error is smaller than the tolerance
            # addition of the computed approximations to get the next value of the function
            #t_values = np.append(t_values, (t + sim_params.step_size))
            t_values = np.append(t_values, (t + dt))
            mass_values = np.append(mass_values, [m], axis = 0)
            pos_values = np.append(pos_values, [p], axis = 0)
            Mdot_values = np.append(Mdot_values, [mdot], axis = 0)
            Rdot_values = np.append(Rdot_values, [rdot], axis = 0)
            filter_values = np.append(filter_values, [ff], axis = 0)
            planet_flux_values = np.append(planet_flux_values, [F_p], axis = 0)
            F0_values = np.append(F0_values, [F0], axis = 0)
            flux_ratio_values = np.append(flux_ratio_values, [F_ratio], axis = 0)

            #debug quantities
            r_acc = np.append(r_acc, [racc], axis = 0)
            h_peb = np.append(h_peb, [hpeb], axis = 0)
            r_acc_h = np.append(r_acc_h, [raccH], axis =0)
            r_acc_b = np.append(r_acc_b, [raccB], axis =0)
            mdot_twod_bondi = np.append(mdot_twod_bondi, [mdottwoDB], axis =0)
            mdot_twod_hill = np.append(mdot_twod_hill, [mdottwoDH], axis =0)
            mdot_threed_bondi = np.append(mdot_threed_bondi, [mdotthreeDB], axis =0)
            mdot_threed_hill = np.append(mdot_threed_hill, [mdotthreeDH], axis =0)
            mdot_threed_unif = np.append(mdot_threed_unif, [mdotthreeDunif], axis =0)
            sigma_peb = np.append(sigma_peb, [sigmapeb], axis=0)
            sigma_gas = np.append(sigma_gas, [sigmagas], axis=0)
            
    simulation = SimulationResults(t_values, mass_values.T, pos_values.T, Mdot_values.T, Rdot_values.T, filter_values.T, 
                                   planet_flux_values.T, F0_values.T, flux_ratio_values.T, r_acc.T, h_peb.T, r_acc_h.T, 
                                   r_acc_b.T, mdot_twod_bondi.T,mdot_twod_hill.T, mdot_threed_bondi.T, mdot_threed_hill.T, 
                                   mdot_threed_unif.T, sigma_peb.T, sigma_gas.T, acc_regimes)
    # Write the result to a JSON file
    with open('sims/sim_'+str(params.H_r_model)+'_'+str(params.v_frag.value)+'.json', 'w') as file:
        json.dump(simulation.__dict__, file, cls=SimulationEncoder)

    return simulation


#for the adaptive timestep
def calculate_error(m, m_new, p, p_new):
    return np.maximum(np.abs(m_new - m) / np.abs(m), np.abs(p_new - p) / np.abs(p)).max()
"""