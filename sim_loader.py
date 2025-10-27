import numpy as np
import multiple_planets_gas_acc as code_gas
from functions_pebble_accretion import *
from functions import *
import functions_plotting as plot
import matplotlib.pyplot as plt
import matplotlib as mpl
import astropy.units as u
import pandas as pd
import json
import os
import re


# functions that construct the simulation results, simulation parameters, and parameters from JSON files
def JSONtoSimRes(filename):
    with open(filename, 'r') as file:
        data = json.load(file)

    simulation_results = code_gas.SimulationResults(
        time=np.array(data['time']) * u.Myr,
        mass=np.array(data['mass']) * u.M_earth,
        position=np.array(data['position']) * u.au,
        dM_dt=np.array(data['dM_dt']) * (u.M_earth / u.Myr),
        dR_dt=np.array(data['dR_dt']) * (u.au / u.Myr),
        filter_fraction=np.array(data['filter_fraction']),
        flux_on_planet=np.array(data['flux_on_planet']),
        F0=np.array(data['F0']),
        flux_ratio=np.array(data['flux_ratio']),
        sigma_peb=np.array(data['sigma_peb']),
        sigma_gas=np.array(data['sigma_gas']),
        acc_regimes=np.array(data['acc_regimes']),
        gas_acc_dict=data['gas_acc_dict']
    )
    
    return simulation_results

def load_params(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    
    # Get the expected keys from the Params class
    expected_keys = set(code_gas.Params.__init__.__code__.co_varnames)
    
    # Filter the JSON data to include only the expected keys
    filtered_data = {k: v for k, v in data.items() if k in expected_keys}
    
    return code_gas.Params(**filtered_data)

def load_sim_params(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    
    # Get the expected keys from the SimulationParams class
    expected_keys = set(code_gas.SimulationParams.__init__.__code__.co_varnames)
    
    # Filter the JSON data to include only the expected keys
    filtered_data = {k: v for k, v in data.items() if k in expected_keys}
    
    return code_gas.SimulationParams(**filtered_data)

# Extract the initial time from the filenames using regular expressions (needed to sort the lists)
def extract_initial_time(filename):
    match = re.search(r't0_(\d+\.\d+)_N_steps', filename)
    if match:
        return float(match.group(1))
    else:
        raise ValueError(f"Filename {filename} does not match the expected format")
