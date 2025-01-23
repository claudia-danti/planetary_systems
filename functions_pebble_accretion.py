import numpy as np
import astropy.units as u
import astropy.constants as const
import matplotlib.pyplot as plt
from functions import *

### UNITS AND CONVERSIONS ######
# masses -> earth masses
# times -> Myr
# lengths -> au
m_kg_to_au_M_E = (1*u.m**2/u.kg).to(u.au**2/u.M_earth).value

## PEBBLE ACCRETION FUNCTIONS ###
def M_dot_ThreeD_unif(position, t, mass, sigma_peb, St, params):
    """Three D accretion for both regimes using t_enc = t_sett, according to Ormel chapter"""
    test = 6*np.pi*R_H(position, mass, params)**3*St*omega_k(position, params)*sigma_peb/(np.sqrt(2*np.pi)*H_peb(St, position, t, params))
    print("3d unif", test)
    return test
     
def M_dot_ThreeD(position, t, R_acc, sigma_peb, v_acc, St, params):
    #Three D accretion according to (A.7) of LM19
    print("mdot three D ", np.pi*R_acc**2*v_acc*sigma_peb/(np.sqrt(2*np.pi)*H_peb(St, position,t, params)))
    return np.pi*R_acc**2*v_acc*sigma_peb/(np.sqrt(2*np.pi)*H_peb(St, position,t, params))

def M_dot_TwoD(R_acc, sigma_peb, v_acc):
    #Two D accretion according to (A.7) of LM19
    print("mdot two D ", 2*R_acc*v_acc*sigma_peb)

    return 2*R_acc*v_acc*sigma_peb

def check_threeD_twoD(position, t, mass, params):
    # checking whether a planet is in 3D or 2D Hill accretion
    if params.St_const == None:
        stokes_peb = st_frag_drift(position, t, params).value

    else: 
        stokes_peb = params.St_const
    if np.pi/(np.sqrt(2*np.pi))*R_H(position, mass, params)*(stokes_peb/2)**(1/3) < H_peb(stokes_peb, position, t, params):
        return True
    else:
        return False


### PEBBLE ACCRETION CLASS ###
class PebbleAccretion:

    def __init__(self, simplified_acc=True):

        self.simplified_acc = simplified_acc
        self.planet_id = None
        self.acc_regime_dict = {}


    def _set_planet_id(self, new_planet_id):

        # Update the planet_id and create new regime times if necessary
        if self.planet_id != new_planet_id:
            self.planet_id = new_planet_id

    
    
    def create_dict_planet_entry(self, planet_id):
      # Create a new entry for the planet with default values
        
        if planet_id not in self.acc_regime_dict:
            # we create two different dictionaries depending whether we use the accretion 3D/2D Hill or full cases
            if self.simplified_acc:
                self.acc_regime_dict[planet_id] = {
                    "masses": {
                        "3D accretion": 0,
                        "2D Hill accretion": 0,

                    },
                    "times": {
                        "3D accretion": 0,
                        "2D Hill accretion": 0,

                    },

                    "printed_2D": False,
                    "printed_3D": False
                }
            else:
                self.acc_regime_dict[planet_id] = {
                    "masses": {
                        "3D accretion": 0,
                        "Bondi accretion": 0,
                        "2D accretion": 0,
                        "Hill accretion": 0
                    },
                    "times": {
                        "3D accretion": 0,
                        "Bondi accretion": 0,
                        "2D accretion": 0,
                        "Hill accretion": 0
                    },
                    
                    "printed_2D": False,
                    "printed_3D": False,
                    "printed_Hill": False,
                    "printed_Bondi": False

                }


    def flagging(self, is_2D_case, is_Hill_case, t, m):

        acc_reg = self.acc_regime_dict[self.planet_id]

        # depending on the accretion regime we are using
        if self.simplified_acc:
            # Check if 2D or 3D accretion has been printed
            for key in ["masses", "times"]:
                for sub_key in ["3D accretion", "2D Hill accretion"]:
                    if acc_reg[key][sub_key] == 0:
                    #if self.first_print_planet:
                        if (is_2D_case and not acc_reg["printed_2D"]) or (not is_2D_case and not acc_reg["printed_3D"]):
                            if is_2D_case:
                                print("Planet "+str(self.planet_id)+" in 2D Hill accretion")
                                acc_reg["times"]["2D Hill accretion"] = t
                                acc_reg["masses"]["2D Hill accretion"] = m
                                acc_reg["printed_2D"] = True

                            else:
                                print("Planet "+str(self.planet_id)+" in 3D accretion")
                                acc_reg["times"]["3D accretion"] = t
                                acc_reg["masses"]["3D accretion"] = m
                                acc_reg["printed_3D"] = True
        #the case of all accretion regimes
        else:
            for key in ["times","masses"]:
                for sub_key in ["Hill accretion", "Bondi accretion"]:
                    if acc_reg[key][sub_key] == 0:
                        if (is_Hill_case and not acc_reg["printed_Hill"]) or (not is_Hill_case and not  acc_reg["printed_Bondi"]):
                            if is_Hill_case:
                                print("Planet "+str(self.planet_id)+" in Hill accretion")
                                acc_reg["times"]["Hill accretion"] = t
                                acc_reg["masses"]["Hill accretion"] = m
                                acc_reg["printed_Hill"] = True
                            else:
                                print("Planet "+str(self.planet_id)+" in Bondi accretion")
                                acc_reg["times"]["Bondi accretion"] = t
                                acc_reg["masses"]["Bondi accretion"] = m
                                acc_reg["printed_Bondi"] = True

                for sub_key in ["2D accretion", "3D accretion"]:
                    if acc_reg[key][sub_key] == 0:
                        if (is_2D_case and not acc_reg["printed_2D"]) or (not is_2D_case and not acc_reg["printed_3D"]):
                            if is_2D_case:
                                print("Planet "+str(self.planet_id)+" in 2D accretion")
                                acc_reg["times"]["2D accretion"] = t
                                acc_reg["masses"]["2D accretion"] = m
                                acc_reg["printed_2D"] = True
                            else:
                                print("Planet "+str(self.planet_id)+" in 3D accretion")
                                acc_reg["times"]["3D accretion"] = t
                                acc_reg["masses"]["3D accretion"] = m
                                acc_reg["printed_3D"] = True
                
    
    def compute_accretion_regime(self, t, position, M_core, St, sigma_peb, sigma_gas, params):

        if self.simplified_acc:
            #simplified case in which I consider ust 3D or 2D Hill
            
            #transition 3D unified acccretion vs 2D Hill accretion
            #if check_threeD_twoD(position, M_core, params):
            #if M_core < M_3D_2DH_trans(position, St, params):
            if np.pi/(np.sqrt(2*np.pi))*R_H(position, M_core, params)*(4*St)**(1/3) < 2*H_peb(St, position, t, params):
                #3D accretion 
                dMc_dt = M_dot_ThreeD_unif(position, t, M_core, sigma_peb, St, params)
                is_2D_case = False
                is_Hill_case = False
                self.flagging(is_2D_case, is_Hill_case, t, M_core)


            else:
                #2D Hill accretion
                dMc_dt = M_dot_TwoD(b_H(position, M_core, St, params), sigma_peb, 3/2*omega_k(position, params)*b_H(position, M_core, St, params))
                is_2D_case = True
                is_Hill_case = True
                self.flagging(is_2D_case, is_Hill_case, t, M_core)

            R_acc = 0
            R_acc_B = b_B(position, M_core, v_hw(position, t, params), St, params)
            R_acc_H = b_H(position, M_core, St, params)


        else:
            #accretion that takes all 4 regimes into account
            #if M_core < M_Bondi_Hill_trans(position, St, params):
            if np.any(v_hw(position, t, params) > 3/2*omega_k(position, params)*b_H(position, M_core, St, params)):
                #Bondi regime
                v_acc = v_hw(position, t, params)
                R_acc = b_B(position, M_core, v_acc, St, params)
                R_acc_B = b_B(position, M_core, v_acc, St, params)
                R_acc_H = 0
                is_Hill_case = False
                print("in Bondi regime", v_acc, R_acc, R_acc_B, R_acc_H)
                is_2d_case = False
                #self.flagging(is_2d_case, is_Hill_case, t, M_core)
                #print("we are in Bondi regime", M_core/M_Bondi_Hill_trans(position, delta_v, params))

            else:
                #Hill regime
                v_acc = 3/2*omega_k(position, params)*b_H(position, M_core, St, params)
                R_acc = b_H(position, M_core, St, params)     
                R_acc_B = 0
                R_acc_H = b_H(position, M_core, St, params)
                is_Hill_case = True
                print("in Hill regime", v_acc, R_acc, R_acc_B, R_acc_H)

                is_2d_case = False
                #self.flagging(is_2d_case, is_Hill_case, t, M_core)
                #print("We are in Hill regime", M_core/M_Bondi_Hill_trans(position, delta_v, params))

            # per il mass trans qua andrebbero fatti tutti gli if del caso considerando quale regime
            # o ci fidiamo che checkare Bondi Hill prima vada bene
            if np.any(R_acc < (2*np.sqrt(2*np.pi)/np.pi)*H_peb(St, position, t, params)):
                #3D accretion
                dMc_dt = M_dot_ThreeD_unif(position, t, M_core, sigma_peb, St, params)
                #dMc_dt = M_dot_ThreeD(position, R_acc, sigma_peb_filtered, v_acc, stokes_peb, params)
                is_Hill_case = False
                print("in 3D regime", dMc_dt)

                is_2d_case = False
                #self.flagging(is_2d_case,is_Hill_case, t, M_core)

            else:
                #2D accretion
                dMc_dt = M_dot_TwoD(R_acc, sigma_peb, v_acc)
                is_Hill_case = False
                print("in 2D regime", dMc_dt)
                is_2d_case = True
                #self.flagging(is_2d_case,is_Hill_case, t, M_core)

        ##debug quantities
        Mdot_twoD_Bondi = M_dot_TwoD(b_B(position, M_core, v_hw(position, t, params), St, params), sigma_peb, v_hw(position, t, params))
        Mdot_twoD_Hill =  M_dot_TwoD(b_H(position, M_core, St, params), sigma_peb, 3/2*omega_k(position, params)*b_H(position, M_core, St, params))
        Mdot_threeD_Bondi = M_dot_ThreeD(position, t, b_B(position, M_core, v_hw(position, t, params), St, params), sigma_peb, v_hw(position, t, params), St, params)
        Mdot_threeD_Hill = M_dot_ThreeD(position, t, b_H(position, M_core, St, params), sigma_peb, 3/2*omega_k(position, params)*b_H(position, M_core, St, params), St, params)
        Mdot_threeD_unif = M_dot_ThreeD_unif(position, t, M_core, sigma_peb, St, params)
        H_r = H_R(position, t, params)
        return dMc_dt, R_acc, H_peb(St, position, t, params)*(2*np.sqrt(2*np.pi)/np.pi), R_acc_H, R_acc_B, Mdot_twoD_Bondi, Mdot_twoD_Hill, Mdot_threeD_Bondi, Mdot_threeD_Hill, Mdot_threeD_unif, sigma_peb, sigma_gas, H_r, self.acc_regime_dict   

    def dMc_dt_f(self, t, M_core, position, flux, flux_reduction_factor, params, sim_params):
        #core accretion rate according to equation (28) of LJ14, filtering fraction passed as an argument

        flux_reduced = flux_reduction_factor*flux
        
        if params.St_const is None:
            stokes_peb = st_frag_drift(position, t, params)
        else: 
            stokes_peb = params.St_const

        sigma_gas = sigma_gas_steady_state(position, t, params)

        #stokes_peb = np.where(position < params.iceline_radius, 1/2, 1)*stokes_peb # to reduce the stokes number at the iceline
        # filtered amount of pebbles to be accreted
        sigma_peb_filtered = sigma_from_flux_general(position, t, flux_reduced, stokes_peb, params)

        dMc_dt, R_acc, Hpeb, R_acc_H, R_acc_B, Mdot_twoD_Bondi, Mdot_twoD_Hill, Mdot_threeD_Bondi, Mdot_threeD_Hill, Mdot_ThreeD_unif, sigma_peb, sigma_gas, H_r, acc_reg_dict  =  self.compute_accretion_regime(t, position, M_core, stokes_peb, sigma_peb_filtered, sigma_gas, params)
        
        print("in dMc_dt", dMc_dt, R_acc, Hpeb, R_acc_H, R_acc_B, Mdot_twoD_Bondi, Mdot_twoD_Hill, Mdot_threeD_Bondi, Mdot_threeD_Hill, Mdot_ThreeD_unif, sigma_peb, sigma_gas, H_r)

        return dMc_dt, R_acc, Hpeb, R_acc_H, R_acc_B, Mdot_twoD_Bondi, Mdot_twoD_Hill, Mdot_threeD_Bondi, Mdot_threeD_Hill, Mdot_ThreeD_unif, sigma_peb, sigma_gas, H_r, acc_reg_dict

################## GAS ACCRETION FUNCTIONS ####################
def M_dot_gas_KH( M_core, params):
    """Kevin-Helmholtz envelope contraction from equation (52) in Nerea's paper"""
    return 10**(-5)*(M_core/10)**4*(params.kappa/0.1*m_kg_to_au_M_E)**(-1)

def M_dot_gas_runaway(position, M_core, t, sigma_gas, params):
    """Runaway gas accretion from equation (53) in Nerea's paper"""
    return 0.29*H_R(position, t, params)**(-2)*(M_core/params.star_mass)**(4/3)*sigma_gas*position**2*omega_k(position, params)*sigma_gap_sigma_gas(position, M_core,t,params)


class GasAccretion:
    def __init__(self):
        self.planet_id = None
        self.gas_accretion_dict = {}

    def _set_planet_id(self, new_planet_id):

        # Update the planet_id and create new regime times if necessary
        if self.planet_id != new_planet_id:
            self.planet_id = new_planet_id

    
    def create_dict_planet_entry(self, planet_id):
      # Create a new entry for the planet with default values
        
        if planet_id not in self.gas_accretion_dict:
            # we create two different dictionaries depending whether we use the accretion 3D/2D Hill or full cases
            self.gas_accretion_dict[planet_id] = {
                "masses": {
                    "KH envelope contraction": 0,
                    "runaway gas accretion": 0,
                    "disc gas accretion": 0,

                },
                "times": {
                    "KH envelope contraction": 0,
                    "runaway gas accretion": 0,
                    'disc gas accretion': 0,

                },

                "printed_KH": False,
                "printed_gas": False,
                "printed_runaway": False,
            }
 
    def flagging(self, is_KH_case, is_runaway_case, is_disc_case, t, m):

        gas_acc = self.gas_accretion_dict[self.planet_id]
        # Check if KH or runaway or disc accretion has been printed
        for key in ["masses", "times"]:
            for sub_key in ["KH envelope contraction", "runaway gas accretion"]:
                if gas_acc[key][sub_key] == 0:
                #if self.first_print_planet:
                    if (is_KH_case and not gas_acc["printed_KH"]) or (not is_KH_case and not gas_acc["printed_runaway"])or (not is_KH_case and not gas_acc["printed_gas"]):
                        if is_KH_case:
                            print("Planet "+str(self.planet_id)+" in KH envelope contraction")
                            gas_acc["times"]["KH envelope contraction"] = t
                            gas_acc["masses"]["KH envelope contraction"] = m
                            gas_acc["printed_KH"] = True

                        elif is_runaway_case:
                            print("Planet "+str(self.planet_id)+" in runaway gas accretion")
                            gas_acc["times"]["runaway gas accretion"] = t
                            gas_acc["masses"]["runaway gas accretion"] = m
                            gas_acc["printed_runaway"] = True
                        else:
                            print("Planet "+str(self.planet_id)+" in disc gas accretion")
                            gas_acc["times"]["disc gas accretion"] = t
                            gas_acc["masses"]["disc gas accretion"] = m
                            gas_acc["printed_gas"] = True

    def dMc_dt_gas(self, t, M_core, position, params, sim_params):

        KH_accretion = M_dot_gas_KH(M_core, params)
        print("KH accretion", KH_accretion)
        runaway_accretion = M_dot_gas_runaway(position, M_core, t, sigma_gas_steady_state(position, t, params), params)
        print("runaway accretion", runaway_accretion)
        mdot_gas_disc = M_dot_star_t(t)
        print("disc accretion", mdot_gas_disc)

        # Determine the minimum accretion rate
        min_accretion = np.minimum(np.minimum(KH_accretion, runaway_accretion), mdot_gas_disc)

        # Set the accretion mode flag
        if min_accretion == KH_accretion:
            accretion_mode = "KH envelope contraction"
            is_KH_case = True
            is_runaway_case = False
            is_disc_case = False

            self.flagging(is_KH_case, is_runaway_case, is_disc_case, t, M_core)

        elif min_accretion == runaway_accretion:
            accretion_mode = "runaway gas accretion"
            is_KH_case = False
            is_runaway_case = True
            is_disc_case = False

            self.flagging(is_KH_case, is_runaway_case, is_disc_case, t, M_core)

        else:
            accretion_mode = "disc"
            is_KH_case = False
            is_runaway_case = False
            is_disc_case = True
            self.flagging(is_KH_case, is_runaway_case, is_disc_case, t, M_core)


        return min_accretion, self.gas_accretion_dict



def dR_dt(t, position, M_core, sigma_g, params):
	"""type one migration according to equation 36 of LJ14 """
	 
	c = 2.8
	dr_dt = -c*M_core/(params.star_mass**2)*sigma_g*position**2*H_R(position, t, params)**(-2)*v_k(position, params)
	return dr_dt

def dR_dt_both(t, position, M_core, sigma_g, params):
    """type one migration according to equation 36 of LJ14 and equation 51 of nerea's paper"""
     
    c = 2.8
    dr_dt_typeI = -c*M_core/(params.star_mass**2)*sigma_g*position**2*H_R(position, t, params)**(-2)*v_k(position, params)
    dr_dt_typeII = dr_dt_typeI*sigma_gap_sigma_gas(position, M_core, t, params)

    if np.any(M_core) < np.any(2.3*M_peb_iso(position, t, params)):
        return dr_dt_typeI
    else:
        return dr_dt_typeII