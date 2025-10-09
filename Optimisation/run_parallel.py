"""
Main script for running twobox optimisation on multiple computer cores.

How to run:
    Set the figure of merit for optimisation in the parameters.py module
    
    Set the parameters and hyperparameters for the global and local optimisation 
    in the parameters.py module
    
    Set the number of computer cores to use in parallel during optimisation
    in the parameters.py module

    Set the maximum number of function evaluations per core and/or the
    maximum runtime (in minutes) per core in the parameters.py module
"""

student = "Shaurya/Test"  # Change this to your name or preferred folder name

# IMPORTS ################################################################################################################################################
import os
## Limit number of numpy threads (MUST GO BEFORE NUMPY IMPORT) ##
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 

from datetime import datetime
from multiprocess import Pool

import numpy as np
from numpy import *

import pathlib
import dill as pickle

import sys
sys.path.append("../")

import fom
import opt 
import parameters
from parameters import FOMSettings, OptimisationSettings, Hyperparameters, Bounds


# Extract settings from parameters.py
choose_monofom, choose_multifom, fom_kwargs = FOMSettings()
num_cores, maxtime, maxstop, runID, xtol_rel, ftol_rel, seed, sampling, n_sample_exp, n_sample = OptimisationSettings()
wavelength, angle, Nx, nG, Qabs, goal, final_speed, return_grad, RCWA_engine, torcwa_sharpness, fixed_parameters = Hyperparameters()
h1_min, h1_max, param_bounds = Bounds()


# RECORDING RESULTS ###########################################################################
# Parameters and hyperparameters for an optimisation run are recorded in dictionaries and saved to a text file for 
# convenient viewing. 

## Converting non-h1 parameter dicts to strings ##
# Fixed parameters
print(f"Multi-wavelength FOM: {choose_multifom}\nMono-wavelength FOM: {choose_monofom}\n", f"FOM kwargs: {fom_kwargs}\n")
hyperparams_dict = {'multifom': choose_multifom, 'monofom': choose_monofom, 'FOM kwargs': fom_kwargs, 
                    'wavelength': wavelength, 'angle': angle, 'Nx': Nx, 'nG': nG, 'Qabs': Qabs,
                    'RCWA engine': RCWA_engine, 'TORCWA edge sharpness': torcwa_sharpness,
                    'Fixed parameters': fixed_parameters}
hyperparams_line = str(hyperparams_dict)
FOM_params_dict = {'final_speed': final_speed, 'goal': goal}
FOM_params_line = str(FOM_params_dict)

# Bounded parameters
bounds_dict = {'param_bounds': param_bounds}
bounds_line = str(bounds_dict)

# Optimiser options
sampling_dict = {'Sampling method': sampling, 'n_sample': f'2E+{n_sample_exp}', 'seed': seed}
sampling_line = str(sampling_dict)
LO_dict = {'xtol_rel': f"{xtol_rel:.1E}", 'ftol_rel': f"{ftol_rel:.1E}"}
LO_line = str(LO_dict)
GO_dict = {'number of cores': num_cores, 'maxstop per core': maxstop}
GO_line = str(GO_dict)

# Date and time at beginning of run
time_at_execution = str(datetime.now())

# Strings to write to file
lines_to_file = ["\n\n------------------------------------------------------------------------------------------------------------------------------------\n"
                , f"Date & time      | {time_at_execution}\n"
                ,  "\n"
                , f"Hyperparameters  | {hyperparams_line}\n"
                , f"FOM parameters   | {FOM_params_line}\n"
                , f"Non-h1 bounds    | {bounds_line}\n"
                ,  "\n"
                , f"Sampling options | {sampling_line}\n"
                , f"LO options       | {LO_line}\n"
                , f"GO options       | {GO_line}\n"
                , "------------------------------------------------------------------------------------------------------------------------------------\n"]


## Writing to file ##
current_dir = pathlib.Path(__file__).resolve(strict=True).parent
txt_fname = f'{runID}_FOM_optimisation_maxtime{maxtime}.txt'
txt_dir = current_dir / "Data" / student / txt_fname

#added this check to make directory if it doesn't exist
#make sure to change "Shaurya" to your own name or preferred folder name
if not os.path.exists(current_dir / "Data" / student):
    os.makedirs(current_dir / "Data" / student)

with open(txt_dir, "a") as result_file:
    result_file.writelines(lines_to_file)



### RUN GLOBAL OPTIMISATION ###########################################################################
# The parallel optimisation is run by partitioning the h1 parameter range into a number (num_cores) of non-intersecting 
# subsidiary parameter ranges whose union is the full h1 parameter range set by the user. Each core optimises over one of 
# those subsidiary h1 ranges, and saves the found most-optimal grating and all optimisation parameters into a dictionary 
# that is stored in .pkl file. The optimisation results for all cores are stored in the same .pkl file.
def optimise_partitioned_depth(h1_bounds):
    _param_bounds = param_bounds[:]
    _param_bounds[0 if "grating_pitch" in parameters.fixed_parameters else 1] = tuple([*h1_bounds])  # Must unpack a single argument for pool.imap to be applied correctly
    return opt.global_optimise(fom.multifom, Hyperparameters(), sampling, seed, n_sample, maxstop, xtol_rel, ftol_rel, _param_bounds)

h1_bounds = []
h1s = np.linspace(h1_min,h1_max,num_cores+1)
for p in range(0,num_cores):
    interval = (h1s[p], h1s[p+1])
    h1_bounds.append(interval)

# Run parallel optimisation
if __name__ == '__main__':
    with Pool(processes=num_cores) as pool:        
        print("Begun!")
        
        # Some processes run for too long, so we need to store the results of each process 
        # immediately once they become available.
        # From https://stackoverflow.com/questions/70317903/how-to-store-all-the-output-before-multiprocessing-finish
        for opt_index, opt_result in enumerate(pool.imap_unordered(optimise_partitioned_depth, h1_bounds)):
            
            opt_FOM = opt_result[0]
            opt_grating = opt_result[1]
            opt_params = opt_result[2]
            is_opt = opt_result[3]
            num_fev = opt_result[4]

            time_at_completion = str(datetime.now())

            data = {'Optimised grating': opt_grating, 'FOM': opt_FOM, 'Real optimum?': is_opt,
                    'Optimised parameters': opt_params, 'Function evaluations': num_fev,
                    'FOM parameters': FOM_params_dict,  'Bounds': bounds_dict,
                    'Sampling settings': sampling_dict, 'LO settings': LO_dict, 'GO settings': GO_dict,
                    'Execution time': time_at_execution, 'Completion time': time_at_completion}
            
            pkl_fname = f'{runID}_FOM_optimisation_maxtime{maxtime}_process{opt_index}.pkl'
            pkl_dir = current_dir / "Data" / student /pkl_fname
            with open(pkl_dir, 'wb') as data_file:
                pickle.dump(data, data_file)
