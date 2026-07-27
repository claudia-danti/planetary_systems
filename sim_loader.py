import numpy as np
import multiple_planets_gas_acc as code_gas
from functions_pebble_accretion import *
from functions import *
import functions_plotting as plot
import matplotlib.pyplot as plt
import matplotlib as mpl
import astropy.units as u
import pandas as pd
import h5py
import os
import re


# ============================================================
# HDF5 LOADING FUNCTIONS
# ============================================================

def HDF5toSimRes(filename):

    with h5py.File(filename, "r") as f:

        sim_grp = f["simulation"]

        data = {}

        for key in sim_grp.keys():

            item = sim_grp[key]

            if isinstance(item, h5py.Dataset):

                value = item[()]

                if "unit" in item.attrs:
                    value = value * u.Unit(item.attrs["unit"])

                data[key] = value

            elif isinstance(item, h5py.Group):

                subdict = {}
 
                for subkey in item.keys():
                    print(type(subkey), repr(subkey), type(item[subkey]))            


                for attr, value in item.attrs.items():
                    subdict[attr] = None if value == "None" else value

                data[key] = subdict

        for attr, value in sim_grp.attrs.items():
            data[attr] = None if value == "None" else value

    expected_keys = set(
        code_gas.SimulationResults.__init__.__code__.co_varnames
    )

    filtered_data = {
        k: v for k, v in data.items()
        if k in expected_keys
    }

    return code_gas.SimulationResults(**filtered_data)


def load_params(filename):

    with h5py.File(filename, "r") as f:

        grp = f["params"]

        data = {
            k: None if v == "None" else v
            for k, v in grp.attrs.items()
        }

    expected_keys = set(
        code_gas.Params.__init__.__code__.co_varnames
    )

    filtered_data = {
        k: v for k, v in data.items()
        if k in expected_keys
    }

    return code_gas.Params(**filtered_data)


def load_sim_params(filename):

    with h5py.File(filename, "r") as f:

        grp = f["sim_params"]

        data = {
            k: None if v == "None" else v
            for k, v in grp.attrs.items()
        }

        for key in grp.keys():
            data[key] = grp[key][()]

    expected_keys = set(
        code_gas.SimulationParams.__init__.__code__.co_varnames
    )

    filtered_data = {
        k: v for k, v in data.items()
        if k in expected_keys
    }

    return code_gas.SimulationParams(**filtered_data)

