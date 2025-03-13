import numpy as np
import astropy.units as u
import astropy.constants as const
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
### UNITS AND CONVERSIONS ######
# masses -> earth masses
# times -> Myr
# lengths -> au
G = (const.G.cgs.to(u.au**3 / (u.M_earth * u.Myr**2))).value #approx 1e8
M_sun_M_E = (const.M_sun.to(u.M_earth)).value
L_sun_au_M_E_Myr = (const.L_sun.cgs).to(u.au**2*u.M_earth/u.Myr**3).value
M_sun_yr_to_M_E_Myr = (1*u.M_sun/u.yr).to(u.M_earth/u.Myr).value #approx 3e12
m_s_to_au_Myr = (1*u.m/u.s).to(u.au/u.Myr).value #approx 210
yr_to_Myr = (1*u.yr).to(u.Myr).value
g_cm2_to_M_E_au2 = (1*u.g/u.cm**2).to(u.M_earth/u.au**2).value
g_cm3_to_M_E_au3 = (1*u.g/u.cm**3).to(u.M_earth/u.au**3).value
mm_to_au = (1*u.mm).to(u.au).value
k_B = const.k_B.to(u.m**2*u.kg/u.s**2/u.K).value

@u.quantity_input

def omega_k(position, params) :
    """ Keplerian frequency """
    return np.sqrt(G * params.star_mass / (position) ** 3)

def v_k(position, params):
    """ Keplerian velocity """
    return np.sqrt(G * params.star_mass / (position))

def v_hw(position, t, params):
    """Headwind velocity experienced by the pebbles"""
    return eta(position, t, params)*v_k(position, params)

def v_H (position, mass, St, params):
    """Hill velocity"""
    return 3/2*omega_k(position, params)*b_H(position, mass, St, params)

def R_H(position, mass, params):
    """ Hill's radius"""
    return (mass/(3*params.star_mass))**(1/3)*position

def R_B(mass, v):
    """Bondi radius"""
    return 2*G*mass/(v**2)

def b_H(position, mass, St, params):
    """Impact parameter (aka accretion radius) in Hill regime, according to my calculations of Ormel chapter"""
    return (4*St)**(1/3)*R_H(position, mass, params)

def b_B(position, mass, v, St, params):
    """Impact paramter (aka accretion radius) in Bondi regime"""
    return np.sqrt(v*St/omega_k(position, params)*R_B(mass, v))

def eta(position, t, params):
    """Pressure reduction term"""
    return -1/2*H_R(position, t, params)**2*params.dlnP_dlnR

def v_nu (position, t, St, params):
    """Gas velocity component in the pebble radial drift velocity (see Ida et al. 2016, equation (27), second term)"""
    return (params.alpha*H_R(position, t, params)**2*v_k(position, params))/(1+St**2)

def v_peb (position, t, St, params):
    """Radial drift velocity of the pebble (see Ida et al. 2016, equation (27), first term)"""
    return 2*(St/(St**2 + 1))*eta(position, t, params)*v_k(position, params)

def v_r(position, t, St, params):
    """Pebble radial drift velocity (see Ida et al. 2016, equation (27))"""
    return v_peb (position, t, St, params) + v_nu(position, t, St, params)

def c_s(T, params):
    """Isothermal speed of sound"""
    return np.sqrt(k_B*T/(params.mu*const.m_p.value))*m_s_to_au_Myr
    
def T_mid (position, t, params):
    """reverting H/R = c_s / v_K to get T(R)"""
    return (H_R(position, t, params)*v_k(position, params))**2*params.mu*(const.m_p/const.k_B/m_s_to_au_Myr)

def r_magnetic_cavity(t, params):
    """Position of the magnetic cavity according to equation (5) of Liu et al. 2017"""
    return (params.star_magnetic_field**4*params.star_radius**12/(4*G*params.star_mass*M_dot_star(t, params)**2))**(1/7)

#### ICELINES #####
# The iceline are computed where T_midplane = 170 K, based on the 
def iceline_irr(t, T, params):
    """Exact calculation using (H/R)_irr = c_s/v_K, with (H/R)_irr from Ida et al.2016"""
    c = 0.024*(params.star_luminosity/L_sun_au_M_E_Myr)**(1/7)*(params.star_mass/M_sun_M_E)**(-4/7)
    c_s_ice = c_s(T, params) #water iceline is defined where T=170K
    return (c/c_s_ice*np.sqrt(G*params.star_mass))**(14/3)

def iceline_visc(t, T, params):
    """Exact calculation using (H/R)_visc = c_s/v_K, with (H/R)_visc from pebble notes"""
    c = 0.024*(params.epsilon_el/1e-2)**(1/10)*(params.epsilon_heat/0.5)**(1/10)*(params.alpha/1e-2)**(-1/10)*(params.Z/0.01)**(1/10)*(params.a_gr/(0.1*mm_to_au))**(-1/10)*(params.rho_gr/g_cm3_to_M_E_au3)**(-1/10)*(M_dot_star(t, params)/(1e-8*M_sun_yr_to_M_E_Myr))**(1/5)
    c_s_ice = c_s(T, params) # water iceline is defined where T=170K
    return (c/c_s_ice*np.sqrt(G*params.star_mass))**(20/9)

def iceline(t, T, params):
    """For a mixed disc the iceline is the max between the two"""
    if params.H_r_model == 'irradiated':
        return iceline_irr(t,T, params)
    else:
        return np.maximum(iceline_irr(t,T, params), iceline_visc(t, T, params))

#### STOKES NUMBER OF THE PARTICLE ######
def st_d(position, t, params):
    """Drift limit using t_g = t_drift for a F=ZM_dot, Sigma_gas = M_dot/(3 pi alpha H Omega) and Sigma_peb = F/(2 pi r v_r), using St^2+1 = 1"""
    # N.B.: this one works just if F = Z*M_dot, for a more general F, use the st_drift_gen
    return np.sqrt(3*np.sqrt(3)/8)*np.sqrt(params.epsilon_p*params.Z*params.alpha)*H_R(position, t, params)**(-1)*(-params.dlnP_dlnR**(-1))

def st_drift_epstein(position, t, params):
    """Drift limit using t_g = t_drift for a Sigma_peb = F/(2 pi r v_r), F St independent, using St^2+1 = 1"""
    # N.B.: this is in epstein regime
    sigma_gas = sigma_gas_steady_state(position, t, params)
    F = params.Z*M_dot_star(t, params)

    return np.sqrt((np.sqrt(3)*params.epsilon_p*F)/(32*np.pi*sigma_gas*position*eta(position, t, params)**2*v_k(position, params)))

def st_drift_stokes(position, t,  params):
    """Drift limited stokes number in Stokes regime, assuming F = Z*M_dot_star, st^2+1 =1"""
    lambda_mfp = mfp(position, t, sigma_gas_steady_state(position, t, params), params)
    rho_gas = rho_0(position, t, sigma_gas_steady_state(position, t, params), params)
    sigma_gas = sigma_gas_steady_state(position, t, params)
    H = H_R(position, t, params)*position
    F = params.Z*M_dot_star(t, params)
    return (48*np.pi/np.sqrt(3)*sigma_gas/F*(rho_gas*H/params.rho_gr)**(-1/2)*np.sqrt(lambda_mfp)/(params.epsilon_p*omega_k(position, params))*eta(position, t, params)**2*v_k(position, params)**2)**(-2/3)

##### WARNING: I REALLY DON'T THINK THIS IS NECESSARY TRUE
def st_drift(position, t,  params):
    """returns the drift stokes number in both drag regimes"""
    return np.maximum(st_drift_epstein(position, t, params), st_drift_stokes(position, t, params))

def st_frag(position, t, params):
    """ fragmentation limited stokes number, according to original equation"""
    return 1/3*(params.alpha_frag)**(-1)*(params.v_frag)**2*(H_R(position, t, params))**(-2)*v_k(position, params)**(-2)

def st_frag_drift(position, t, params):
    """Returns the minimum between drift and fragmentation"""
    if params.St_const == None:

        return np.minimum(st_drift(position, t, params), st_frag(position, t, params))
    else: 
        return params.St_const

#### Midplane gas density
def rho_0_t (position, t, params):
    """Midplane gas density assuming isothermal disc"""
    sigma_gas = sigma_gas_steady_state(position, t, params)
    return sigma_gas/(np.sqrt(2*np.pi)*H_R(position, t, params)*position)

def rho_0 (position, t, sigma_gas, params):
    """Midplane gas density assuming isothermal disc"""
    return sigma_gas/(np.sqrt(2*np.pi)*H_R(position, t, params)*position)

def mfp (position, t, sigma_gas, params):
    """Mean free path of the gas"""
    return (params.mu*const.m_p.cgs.value/params.cross_sec_H.value)*g_cm2_to_M_E_au2/rho_0(position, t, sigma_gas, params)

def st_from_r_peb(r_peb, position, t, params):
    """relation between stokes number and pebble dimension, both regimes"""
    lambda_mfp = mfp(position, t, sigma_gas_steady_state(position, t, params), params)
    rho_gas = rho_0(position, t, sigma_gas_steady_state(position, t, params), params)

    #CAVEAT, THE NP.ANY COULD NOT WORK IN THE CODE
    if np.any(r_peb < 9/4*lambda_mfp):
        print("Epstein regime")
        return params.rho_gr*r_peb/(rho_gas*H_R(position, t, params)*position) 
    else:  
        print("Stokes regime")
        return 4/9*r_peb**2*params.rho_gr/(rho_gas*lambda_mfp*H_R(position, t, params)*position)

def r_peb_from_st(st, position, t, params):
    """relation between pebble dimension and stokes number, both regimes"""
    lambda_mfp = mfp(position, t, sigma_gas_steady_state(position, t, params), params)
    rho_gas = rho_0(position, t, sigma_gas_steady_state(position, t, params), params)
    #CAVEAT, THE NP.ANY COULD NOT WORK IN THE CODE

    if np.any(st < 9/4*(params.rho_gr*lambda_mfp)/(rho_gas*H_R(position, t, params)*position)):
        print("Epstein regime")
        return st*rho_gas*H_R(position, t, params)*position/(params.rho_gr)
    else:  
        print("Stokes regime")
        return np.sqrt(9/4*st*lambda_mfp*rho_gas*H_R(position, t, params)*position/(params.rho_gr))

###### TIME DEPENDENT GAS ACCRETION ######
def M_dot_star_t(t):
    #Time dependent gas accretion rate accordin to Hartmann et al. 2016 (ann.rev.), lower limit
    return 10**(((-1.32)-(1.07)*np.log10(t/yr_to_Myr)))*M_sun_yr_to_M_E_Myr

def M_dot_star(t, params):
    """Gas accretion rate, can be time dependent or constant"""
    if params.M_dot_star == None:
        M_dot_star = M_dot_star_t(t)
    else: 
        M_dot_star = params.M_dot_star

    return M_dot_star

###### GAS SURFACE DENSITIES ############
def sigma_gas_irr(position, t, params):
    """Sigma gas for an irradiated disc, equation (13) from Ida et al. 2016"""

    return 2.7*1e3*(params.star_mass/M_sun_M_E)**(9/14)*(params.star_luminosity/L_sun_au_M_E_Myr)**(-2/7)*(M_dot_star(t, params)/(1e-8*M_sun_yr_to_M_E_Myr))*(params.alpha/1e-3)**(-1)*(position)**(-15/14)

def sigma_gas_visc_Ida(position, t, params):
    """Sigma gas for a viscous disc, equation (12) from Ida et al. 2016"""

    return 2.1*1e3*(params.star_mass/M_sun_M_E)**(1/5)*(params.alpha/1e-3)**(-4/5)*(M_dot_star(t, params)/(1e-8*M_sun_yr_to_M_E_Myr))**(3/5)*(position)**(-3/5)

def sigma_gas_visc_Liu(position, t, params):
    """Sigma gas for viscous disc from Liut et al. 2019, equation (8)"""

    return 132*(M_dot_star(t, params)/(1e-8*M_sun_yr_to_M_E_Myr))**(1/2)*(params.star_mass/M_sun_M_E)**(1/8)*(params.alpha/1e-2)**(-3/4)*(params.disc_opacity/1e-2)**(-1/4)*(position)**(-3/8)

def sigma_gas_steady_state(position, t, params):
    """Gas surface density for a steady state viscously evolving disc, equation (11) in Ida et al. 2016"""
    return M_dot_star(t, params)/(3*np.pi*params.alpha*H_R(position, t, params)**2*position**2*omega_k(position, params))

###### PEBBLE SURFACE DENSITIES ############
# this surface density already contains the NON constant Stokes number of the particles (is obtained combining eq (14), (15), (20) and (24))
# So be caerful this intrinsically has a NON CONSTANT Stokes number for the particle
# Other warning, this describes a pebble surface density obtained from a flux described by eq (14), by the function flux_peb_t
def sigma_peb_t(position, t, sigma_gas, params):
    """time dependent pebble surface density as in eq 25 in LJ14"""

    C = 2 ** (5 / 6) * 3 ** (-7 / 12) * ((params.epsilon_d ** (1 / 3)) / (params.epsilon_p ** (1 / 2))) * params.Z ** (5 / 6)
    return C * sigma_gas * omega_k(position, params) ** (-1 / 6) * t ** (-1 / 6)


########### ratio of gap to gas surface density for type II migration ##########
def sigma_gap_sigma_gas(position, mass, t, params):
    """ratio between gap and gas surface density as in Johansen 2019 eq. 37"""
    return 1/(1+(mass/(2.3*M_peb_iso(position, t, params)))**2)

def sigma_gap_sigma_gas_gen(position, mass, t, params):
    """ratio between gap and gas surface density as in Johansen 2019 eq. 35"""
    K = (mass/params.star_mass)**2*H_R(position, t, params)**(-5)/params.alpha
    return 1/(1+0.04*K)


########## FLUXES #############
# This pebble flux is obtained through the computation of the pebble production line
def flux_peb_prod_line(t, position, params):
    """pebble flux as in eq (14) pf LJ14, modified to have a 1/3 reduction when crossing the iceline"""
    
    beta = params.beta0 * np.exp(-t / params.tau_disc)
    C = (2 / 3) ** (2 / 3) * np.pi * (beta * (1 * u.au).to(u.cm)) * params.epsilon_d ** (2 / 3) * params.Z ** (5 / 3)
    flux =  C * (const.G.cgs * params.star_mass) ** (1 / 3) * t ** (-1 / 3)
    F = np.where(position < params.iceline_radius, 1/2, 1)*flux

    return F

# flux from the dust-to-gas ratio, position dependent
def flux_dtg_t(t, params):
    """Flux as a fraction (Z) of the mass accretion rate on the star"""
    return params.Z*M_dot_star(t, params)

def flux_const(t, params, sim_params):

    """Flux designed to make the 5 AU planet reach pebble isolation mass in 2 Myr"""
    #CAVEAT: the initial time and mass are set here, could potentially be different from the simulation!!!
    # WARNING, doesn't handle the iceline, overwrites the alpha parameter with the value of alpha_out
    # handling of the iceline is done in the main routine 

    ## initial parameters of the Jupiter-like planets
    pos = (5*u.au).value
    t_in = (0.1*u.Myr).value## initial time
    t_iso = (2*u.Myr).value# we want the planet to reach peb iso at 2 Myrs    
    M_iso = M_peb_iso(pos, t_iso, params)

    sigma_gas = sigma_gas_steady_state(pos, t_in, params) #used to compute the initial mass of the 5 au planet, needed for the flux

    M0 = M0_pla(pos, t_in, sigma_gas, params) #initial mass of the 5AU embryo using planetesimal imf
    #print("M0", M0.to(u.earthMass))

    # the St and transition mass need to be computed at the location of Jupiter-like planet, to get the targeted flux
    stokes_peb = st_frag_drift(pos, t, params).value

    #print("St in flux", stokes_peb)
    M_trans = M_3D_2DH_trans(pos, t, stokes_peb, params)

    #The f_3D and f_2D contain the (H/r) and alpha dependecies of the accretion equations
    #beware that I used the alpha_z_out parameter for the flux because I want F to be constant also if alpha_z changes
    H_peb_H = H_peb(stokes_peb, pos, t, params)/(pos*H_R(pos, t, params))
    f_3D = 1/(np.sqrt(2*np.pi))*(params.star_mass)**(-1)*(H_R(pos, t, params))**(-1)*(H_peb_H)**(-1)*(stokes_peb**2+1)/(2*eta(pos, t, params))
    f_2D = (3/(4*stokes_peb))**(1/3)*(params.star_mass)**(-2/3)*(stokes_peb**2+1)/(np.pi*eta(pos, t, params))

    #to check the integration limit: is M_transition before or after the isolation mass?

    if M_trans < M_iso:
        F = (np.log(M_trans/M0)/f_3D + 3*(M_iso**(1/3) - M_trans**(1/3))/f_2D)/(t_iso - t_in)
    else:
        F = (np.log(M_iso/M0)/f_3D)/(t_iso - t_in)

    return F


########## CONVERSION FLUX <-> SIGMA #############
############ CAN have constant Stokes number #########
# This function can be used with a constant Stokes number (St passed as an argument)
def flux_from_sigma_general(position, t, sigma_peb, St, params):
    """ pebble flux from pebble surface density using eq (15) and (17, 18) from LJ14 """
    
    return sigma_peb * 2 * np.pi * position * v_r(position, t, St, params)

# This function can be used with a constant Stokes number (St passed as an argument)
def sigma_from_flux_general(position, t, flux, St, params):
    """ function that gets the pebble surface density starting from flux, using equation (15) and (17,18) of LJ14 """

    return flux/(2*np.pi*position*v_r(position, t, St, params))

############ NON constant Stokes number ################
# N.B.: This function intrinsically contains already a NON CONSTANT Stokes number
def flux_from_sigma(position, sigma_peb, sigma_gas, params): #WARNING NOT TESTED
    """ function that gets the flux starting from the pebble surface density according to equation (24) in LJ14 """

    return np.sqrt(3)*np.pi*params.epsilon_p*position*sigma_peb**2*v_k(position, params)/(2*sigma_gas)

# N.B.: This function intrinsically contains already a NON CONSTANT Stokes number
def sigma_from_flux(position, flux, sigma_gas, params):
    """function that gets the pebble surface density starting from the pebble flux, using equation (24) of LJ14"""

    return np.sqrt((2*flux*sigma_gas)/(np.sqrt(3)*np.pi*params.epsilon_p*position*v_k(position, params)))

########## SCALE HEIGHTS #############
def H_R_irr(position, params):
    """Irradiated disc prescription equation (10) in Ida et al. 16"""
    return 0.024*(params.star_luminosity/L_sun_au_M_E_Myr)**(1/7)*(params.star_mass/M_sun_M_E)**(-4/7)*(position)**(2/7)

def H_R_visc_Ida(position, t, params):
    """Viscous disc prescription equation (9) in Ida et al. 16"""

    return 0.027*(params.star_mass/M_sun_M_E)**(-7/20)*(params.alpha/1e-3)**(-1/10)*(M_dot_star(t, params)/(1e-8*M_sun_yr_to_M_E_Myr))**(1/5)*(position)**(1/20)

def H_R_visc_Liu(position, t, params):
    """Viscous H/r from equation (10) in Liu et al. 2019"""
    return 0.034*(params.star_mass/M_sun_M_E)**(-5/16)*(M_dot_star(t, params)/(1e-8*M_sun_yr_to_M_E_Myr))**(1/4)*(params.alpha/1e-2)**(-1/8)*(position)**(-1/16)*(params.disc_opacity/1e-2)**(1/8)

def H_R_visc_Lambrechts(position, t, params):
    """Viscous H/r from equation (A47) in Lambrechts notes"""
    return 0.019*(params.epsilon_el/1e-2)**(1/10)*(params.epsilon_heat/0.5)**(1/10)*(params.alpha/1e-2)**(-1/10)*(params.Z/0.01)**(1/10)*(params.a_gr/(0.1*mm_to_au))**(-1/10)*(params.rho_gr/g_cm3_to_M_E_au3)**(-1/10)*(position)**(1/20)*(M_dot_star(t, params)/(1e-8*M_sun_yr_to_M_E_Myr))**(1/5)*(params.star_mass/M_sun_M_E)**(-7/20)

def H_R_flared(position, params):
    """Flared disc Bitsch et al. 2015"""
    #this is basically the optically thin limit of H/R Ida equations but without M* and L*
    return 0.033 * (position) ** (1 / 4)

def H_R(position, t, params):
    """H/r possible modes, from Ida, Liu, Bitsch"""
    
    if params.H_r_model == "Ida_mixed":
        #print("Ida H/R:", np.maximum(flared_disc_Ida, flat_disc_Ida))
        return np.maximum(H_R_irr(position, params), H_R_visc_Ida(position, t, params))
    
    if params.H_r_model == "Liu_mixed":
        #print("Liu H/R:", np.maximum(flared_disc_Ida, flat_disc_Ida))
        return np.maximum(H_R_irr(position, params), H_R_visc_Liu(position, t, params))
    
    if params.H_r_model == "Lambrechts_mixed":
        return np.maximum(H_R_irr(position, params), H_R_visc_Lambrechts(position, t, params))

    if params.H_r_model == 'irradiated':
        return H_R_irr(position, params)

    if params.H_r_model == "flared":
        #print("Irradiated H/R:", flared_disc)
        return H_R_flared(position, params)

    if params.H_r_model == "flat":
        #print("Viscous H/R:", flat_disc)
        return 0.04


def H_peb(St, position, t, params):
    """pebble scale height according to equation (A.6) of LJ19"""
    H_gas = position*H_R(position, t, params)
    return np.sqrt(params.alpha_z / (St+ params.alpha_z)) * H_gas


######### TRANSITION MASSES, ISOLATION MASS, INITIAL MASS #################
## maybe needs an update with the Bitsch one!
def M_peb_iso(position, t, params):
    """pebble isolation mass according to annual review equations"""
    return 20 * (H_R(position, t, params) / 0.05)**3 *(params.star_mass/M_sun_M_E)

def M_peb_iso_Bitsch(position, t, params):
    """Pebble isolation mass from Bitsch et al. 2018"""
    stokes_peb = st_frag_drift(position, t, params)

    #tecnically there is also the diffusion term, equation (26)
    f_fit = (H_R(position, t, params)/0.05)**3*(0.34*(np.log(0.001)/np.log(params.alpha))**4+0.66)*(1-(params.dlnP_dlnR+2.5)/6)
    Pi_crit = params.alpha_z/(2*stokes_peb)
    l = 0.00476/f_fit
    return 25*f_fit* + Pi_crit/l

def M_Bondi_Hill_trans(position, t, params):
    """3D/2D transition mass according to Ormel review"""
    stokes_peb = st_frag_drift(position, t, params)

    return v_hw(position, t, params)**3/(8*G*omega_k(position, params)*stokes_peb)*(4/3)**2

def M_Bondi_Hill_trans_debug(position, t, stokes, params):
    return (4/3*omega_k(position, params))**2*(eta(position, t, params)*position/2)**3/(stokes*G)

def M_cut(position, t, params):
    """Lower limit for pebble accretion, according to equation (17) in the notes"""
    stokes_peb = st_frag_drift(position, t, params)

    return 1e-6*stokes_peb/1e-2*(eta(position, t, params)/1.8e-3)**3

def M_peb_cutoff(position, t, params):
    """Lower limit for pebble accretion, according to equation (17) in the notes"""
    stokes_peb = st_frag_drift(position, t, params)

    return 2.5*1e-4*(stokes_peb/0.1)*(eta(position, t, params)/0.002)**3*(params.star_mass)


# WARNING I used alpha_z so it could not work when changing alpha across the iceline
def M_3D_2DH_trans(position, t, params):

    stokes_peb = st_frag_drift(position, t, params)

    return 3/4*(2*H_peb(stokes_peb, position, t, params)*(np.sqrt(2*np.pi))/np.pi)**3*(omega_k(position, params)**2/(G*stokes_peb))

def M_3D_2DB_trans (position, t, params):

    stokes_peb = st_frag_drift(position, t, params)

    return (8*H_peb(stokes_peb, position, t, params)**2/np.pi)*eta(position, t, params)*omega_k(position, params)**2*position/(G*stokes_peb)

def M0_pla(position, t, sigma_gas, params):
    """Initial planetesimal mass scaling, according to Liu et al. 2020"""
    f = 400
    f_value = f/400
    return 2e-4*f_value*(H_R(position, t, params)/0.04)**(3/2)*(sigma_gas/(1700*g_cm2_to_M_E_au2))**(3/2)*(position)**3

def pebble_prod_line(time, params):
    """pebble production line according to eq 10 of LJ14"""

    return (3 / 16) ** (1 / 3) * (G * params.star_mass) ** (1 / 3) * (params.epsilon_d * params.Z) ** (2 / 3) * time ** (2 / 3)

def t_acc(mass, F, filter_frac):
    """time to accrete a mass m with a given flux"""
    return mass/(filter_frac*F)

def meanmr_two_one_in(a_p_out):
    """gets the position of the 2:1 mean motion resonance inner planet given an outer planet"""
    return (1/2)**(2/3)*a_p_out

def meanmr_two_one_out(a_p_in):
    """gets the position of the 2:1 mean motion resonance outer planet given an inner planet"""
    return (2)**(2/3)*a_p_in

def estimate_initial_step_size(masses, positions, mdot, rdot):
    # Estimate characteristic timescales
    tau_m = np.abs(masses / mdot)
    tau_p = np.abs(positions / rdot)
    
    # Choose a fraction of the smallest timescale
    initial_step_size = 0.1 * np.min([tau_m.min(), tau_p.min()])
    
    return initial_step_size


def planet_counter(simulations, parameters, sim_parameters):
    """Counts the types of planets in the simulation"""
    HJ_counter, WJ_counter, CG_counter, SE_counter, sub_E_counter, terr_in_counter, giant_counter = 0,0,0,0,0,0,0

    for i in range(len(simulations)):
        sim = simulations[i]
        params = parameters
        sim_params = sim_parameters

        for p in range(1,sim_params.nr_planets):
            idx = idxs (sim.time[p], sim.mass[p], sim.position[p], sim.filter_fraction[p], 
                            sim.dR_dt[p], sim.dM_dt[p], params, True)
            stop_mig_idx = idx['stop_mig_idx'].values[0]

            m_fin_idx = stop_mig_idx

            if 1<sim.mass[p,m_fin_idx].to(u.M_earth).value<20 and sim.position[p,m_fin_idx].to(u.au).value<1:
                SE_counter +=1
            if 0.01<sim.mass[p,m_fin_idx].to(u.M_earth).value<1:
                if sim.position[p, m_fin_idx].to(u.au).value<0.1:
                    terr_in_counter +=1    
                else:
                    sub_E_counter +=1
            if sim.mass[p,m_fin_idx].to(u.M_earth).value>=100:
                giant_counter += 1
                if 0.01<sim.position[p,m_fin_idx].to(u.au).value<0.1:
                    HJ_counter +=1
                if 0.1<sim.position[p,m_fin_idx].to(u.au).value<2:
                    WJ_counter +=1
                if 2<sim.position[p,m_fin_idx].to(u.au).value<10:
                    CG_counter +=1
    dict_planets = {'HJ': HJ_counter, 'WJ': WJ_counter, 'CG': CG_counter+1, 'SE': SE_counter, 'sub_E':sub_E_counter,  'terr_in': terr_in_counter, 'terr_tot': terr_in_counter+sub_E_counter,  'giant': giant_counter}
    return dict_planets


def idxs (time, mass, position, filter_fraction, dR_dt, dM_dt, params, migration, **kwargs):
    #Creates the index dictionary

    idx_or_last = lambda fltr: np.argmax(fltr) if np.any(fltr) else fltr.size
    isolation_mass = M_peb_iso(position.value, time.value, params)
    stop_idx = np.argmin(position) #returns the position of the min value of position
    stop_mass_idx = np.any(np.where(dM_dt == 0)[0][0]) if np.any(np.where(dM_dt == 0)[0]) else dM_dt.size
    isolation_idx = np.where(mass.value > isolation_mass)[0]
    if isolation_idx.size > 0:
        isolation_idx = isolation_idx[0]
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



def kepler_3_law (period):
    return ((const.G.cgs*const.M_sun.cgs/(4*np.pi**2)*period**2)**(1/3)).to(u.au)

def radius_mass_exo(radius, rocky = True):
    if rocky:
        rho = 5.5* u.g/u.cm**3
    else:
        rho = 1* u.g/u.cm**3
    return 4/3*np.pi*radius**3*rho



def MMSN (position):
    """Minimum mass solar nebula according to Hayashi 1981"""
    return (1700*(position)**(-3/2)*u.g/u.cm**2).to(u.M_earth/u.au**2)