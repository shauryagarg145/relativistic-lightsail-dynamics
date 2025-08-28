"""
A module for storing the user-defined figure of merit function and global optimisation function. 

You should import your figure-of-merit functions from opt.py into your main optimisation script. 
"""

# IMPORTS ########################################################################################################################


import numpy as np
import nlopt

import os
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
from operator import itemgetter

import dill as pickle

import sys
sys.path.append("../")

import parameters
I0, L, m, c = parameters.Parameters()
from twobox import TwoBox



## CONSTRAINT FUNCTIONS ##
# Constraints have the form h(x) <= 0, i.e. the constraint function should return a positive 
# value if the constraint is violated. Additionally, MMA takes the gradients of the constraints,
# so the constraint functions should be differentiable with respect to the optimisation parameters. 
# However, I think the gradients of constraint functions are obtained by MMA internally (likely using
# finite differences), so autogradability is not needed.
# Some of these constraints partially overlap with the bound constraints set for the global optimizer,
# but here, we can pass the exact parameters to the constraints rather than predetermining the bounds.
# In the case where the pitch is fixed, we assume that the user has set the bounds accordingly, 
# so we do not add the box1_too_wide, box2_too_wide or bcd_redundant constraints to the local optimiser.
# TODO: How do we know which parameter has which index, since params length is variable?
#       Current fix is to disable certain constraint functions if pitch is fixed during
#       optimisation (see global_optimise()).
def box1_too_wide(params,gradn): 
    """
    Constraint function to prevent box1 from being wider than the unit cell.
    """
    Lam = params[0]  
    w1 = params[2]  
    condition = w1 - Lam 
    return condition

def box2_too_wide(params,gradn): 
    """
    Constraint function to prevent box2 from being wider than the unit cell.
    """
    Lam = params[0]  
    w2 = params[3]
    condition = w2 - Lam 
    return condition

def bcd_redundant(params,gradn): 
    """
    Constraint function containing two conditions to avoid redundant parameter space:
        Symmetry wrt swapping box1 and box2, avoid by taking box centre distance > 0
        Unit cell periodicity means boxes with separation >0.5*Lam are equivalent to boxes with separation <0.5*Lam
    """
    Lam = params[0]  
    bcd = params[4] 
    condition = np.abs(bcd - 0.25*Lam) - 0.25*Lam 
    return condition

def boxes_overlap(params,gradn):
    """
    Constraint function to guarantee an asymmetric unit cell by ensuring the distance between two unit cells is larger than zero
    TODO: The boxes overlapping can still be asymmetric, so this constraint is slightly too restrictive
            However, would need to find a way to handle gradients in the overlap regime.
    """
    w1 = params[2]
    w2 = params[3]
    bcd = params[4]
    condition= (w1+w2)/2 - bcd 
    return condition

def boxes_clip_unit_cell(params,gradn):
    """
    Constraint function that avoids the boxes clipping the edge of the unit cell.
    
    Boxes clipping the unit cell edges can lead to unexpected objective values (user input box widths may not correspond to the 
    box width that RCWA receives). Additionally, gradients become expensive to compute in this case.
    """
    if "grating_pitch" in parameters.fixed_parameters:
        """This constraint must always be active, so we must handle the fixed pitch case."""
        pitch_idx = parameters.fixed_parameters.index("grating_pitch")
        Lam = parameters.fix_parameter_values[pitch_idx]
        w1 = params[1]
        w2 = params[2]
        bcd = params[3]
    else:
        Lam = params[0]
        w1 = params[2]
        w2 = params[3]
        bcd = params[4]
    condition = (w1+w2)/2 + bcd - 0.98*Lam
    return condition


def global_optimise(objective_fom, opt_hyperparams,
                    sampling_method: str="sobol", seed: int=0, n_sample: int=8, maxstop: dict={'maxfev': 1000, 'maxtime': 600},
                    xtol_rel: float=1e-4, ftol_rel: float=1e-8, param_bounds: list=[], return_settings: bool=False):
    """
    Global optimise the twobox on a single CPU core using MLSL global optimiser with internal MMA local optimiser.

    Parameters
    ----------
    objective_fom    :   Figure of merit function to be optimised. Must take a grating object and return a tuple (FOM, gradient).
    opt_hyperparams  :   Hyperparameters for the optimisation 
    sampling_method  :   "sobol" or "random" initial point sampling
    seed             :   Seed for initial random parameter space sample and grating_depth samples
    n_sample         :   Number of points for initial sample (per dimension?)
    maxstop          :   Set maximum function evaluations and/or maximum walltime (minutes) per core
    xtol_rel         :   Relative position tolerance for MMA
    ftol_rel         :   Relative objective tolerance for MMA
    param_bounds     :   Ordered list of tuples, one tuple per parameter bound, each tuple containing one lower and one upper bound.
    return_settings  :   If true, return the optimisation settings. If false, return only the FOM and the grating object.
    """
    
    ndof = len(param_bounds)  # number of optimisation parameters
    init_params = []  # Initial parameters for the optimiser, length determined by non-None param_bounds
    init_all_params = []
    param_bound_idx = 0
    fixed_param_idx = 0
    for param_name in parameters.param_names:
        if param_name not in parameters.fixed_parameters:
            pb = param_bounds[param_bound_idx]
            p_avg = (pb[0]+pb[1])/2
            init_params.append(p_avg)
            init_all_params.append(p_avg)
            param_bound_idx += 1
        else:
            # If the parameter is fixed, use the fixed value from parameters.py
            init_all_params.append(parameters.fix_parameter_values[fixed_param_idx])
            fixed_param_idx += 1

    # Set up the grating object to be updated during optimisation and returned
    wavelength, angle, Nx, nG, Qabs, goal, final_speed, return_grad, RCWA_engine, torcwa_sharpness, fixed_parameters = opt_hyperparams
    grating = TwoBox(*init_all_params, wavelength, angle, Nx, nG, Qabs, RCWA_engine, torcwa_sharpness, fixed_parameters)

    def objective(grating, opt_params):
        """
        Objective function to be optimised. Is likely fom.multifom. 
        However, the monofom kwarg is not passed here since, 
        for optimisation only, the monofom is pre-selected 
        in parameters.py.
        """
        grating.params = opt_params
        return objective_fom(grating, final_speed=final_speed, goal=goal, return_grad=return_grad)

    def fun_nlopt(params,gradn):
        """
        nlopt objective function

        Arguments: the list of parameters and the gradient of the objective with respect to the parameters at the given params.

        Returns: value of the objective function at the given params
        """
        y, dy = objective(grating,params)
        if gradn.size > 0:  # Even for gradient methods, in some calls gradn will be empty []
            gradn[:] = dy
        
        # # Debugging: Print constraint values to ensure optimiser moves to negative regions
        # bcd_red = bcd_redundant(params,gradn)
        # boxes_overl = boxes_overlap(params,gradn)
        # boxes_clip = boxes_clip_unit_cell(params,gradn)
        # print(f"Param: {params}")
        # print(f"Gradn: {gradn}")
        # print(bcd_red)
        # print(boxes_overl)
        # print(boxes_clip)
        # print("\n")
        
        return y

    if sampling_method == 'sobol':
        global_opt = nlopt.opt(nlopt.G_MLSL_LDS, ndof)
    elif sampling_method == 'random':
        global_opt = nlopt.opt(nlopt.G_MLSL, ndof)
    else:
        global_opt = nlopt.opt(nlopt.G_MLSL_LDS, ndof)
    local_opt = nlopt.opt(nlopt.LD_MMA, ndof)

    nlopt.srand(seed) 
    n_sample = int(n_sample)
    global_opt.set_population(n_sample)  # set initial sampling points

    local_opt.add_inequality_constraint(boxes_clip_unit_cell)
    local_opt.add_inequality_constraint(boxes_overlap)
    if "grating_pitch" not in parameters.fixed_parameters:
        """
        If pitch is fixed, we assume that parameter bounds have 
        been set according to the fixed pitch value. E.g. 
        box widths should be less than the grating pitch.
        If pitch is not fixed, we need the following box constraints.
        """
        local_opt.add_inequality_constraint(box1_too_wide)
        local_opt.add_inequality_constraint(box2_too_wide)
        local_opt.add_inequality_constraint(bcd_redundant)

    local_opt.set_xtol_rel(xtol_rel)
    local_opt.set_ftol_rel(ftol_rel)
    global_opt.set_local_optimizer(local_opt)

    global_opt.set_max_objective(fun_nlopt)
    if "maxfev" in maxstop:
        maxfev = maxstop["maxfev"]
    else:
        maxfev = -1  # -1 means no limit
        print("Warning: maxfev not provided in maxstop dictionary. Ignore if this is intentional.")
    if "maxtime" in maxstop:
        maxtime = 60*maxstop["maxtime"]  # Seconds
    else:
        maxtime = -1  # -1 means no limit
        print("Warning: maxtime not provided in maxstop dictionary. Ignore if this is intentional.")
    global_opt.set_maxeval(maxfev)
    global_opt.set_maxtime(maxtime)

    lb = [bounds[0] for bounds in param_bounds]
    ub = [bounds[1] for bounds in param_bounds]
    global_opt.set_lower_bounds(lb)
    global_opt.set_upper_bounds(ub)

    opt_params = global_opt.optimize(init_params)
    optimum = objective(grating,opt_params)[0]
    is_optimum = True
    num_fev = global_opt.get_numevals()
    print("Success on starting bounds: ", param_bounds[1])
    

    if return_settings:
        settings = {"sampling_method": sampling_method, "seed": seed, "n_sample": n_sample,
                    "maxstop": maxstop, "xtol_rel": xtol_rel, "ftol_rel": ftol_rel,
                    "param_bounds": param_bounds}
        return (optimum, grating, opt_params, is_optimum, num_fev, settings)
    else:
        return (optimum, grating, opt_params, is_optimum, num_fev)

def extract_opt(data_basefile_name: str, num_processes: int=8, output_opt_idx: int=0):
    """
    Extract the optimum gratings stored in multiple data files, each file corresponding to the output of an optimisation core. 
    Optima are ordered by FOM (largest to smallest).

    Parameters
    ----------
    data_basefile_name  :   pkl file name relative to working directory
    num_processes       :   Number of processes used in the optimisation that must be individually extracted
    output_opt_idx      :   Index for the optimal twobox instance (from the ordered list) to return directly

    Returns
    -------
    maxima_and_maximisers_sorted :   List of tuples, each tuple being (FOM, optimisation parameters) 
    opt_gratings_sorted          :   List of tuples, each tuple being (FOM, twobox instance)
    chosen_best_grating          :   Twobox instance for the output grating chosen by output_opt_idx
    """

    opt_FOMs = []
    opt_gratings = []
    opt_params = []
    total_fev = 0
    for n in range(num_processes):
        try: 
            data_fname = str(data_basefile_name) + f"_process{n}.pkl"
            with open(data_fname, 'rb') as data_file:
                data = pickle.load(data_file)
        except FileNotFoundError:
            continue

        opt_FOMs.append(data["FOM"])
        opt_gratings.append(data["Optimised grating"])
        opt_params.append(data["Optimised parameters"])

        try:
            total_fev += data["Function evaluations"]
        except KeyError as ke:
            print(ke)

    print(f"Total function evaluations: {total_fev}")
    print(f"Average function evaluations per core: {int(total_fev/num_processes)}")
    maxima_and_maximisers = zip(opt_FOMs, opt_params)
    maxima_and_gratings = zip(opt_FOMs, opt_gratings)

    # Sort the optima based on the FOM value
    maxima_and_maximisers_sorted = sorted(maxima_and_maximisers, key=itemgetter(0), reverse=True)
    opt_gratings_sorted = sorted(maxima_and_gratings, key=itemgetter(0), reverse=True)
    try:
        chosen_best_grating = opt_gratings_sorted[output_opt_idx][1]
    except IndexError:  # TODO: search for the file name directly rather than handling here
        raise FileNotFoundError(f"Warning: Optimisation results were not loaded correctly. Check base filename: {data_basefile_name}")

    return maxima_and_maximisers_sorted, opt_gratings_sorted, chosen_best_grating

def extract_opt_single(data_filename: str, output_opt_idx: int=0):
    """
    Extract the optimum gratings stored in a single data file. Optima are ordered by FOM (largest to smallest).

    Parameters
    ----------
    data_filename  :   pkl file name relative to working directory
    output_opt_idx :   Index for the optimal twobox instance (from the ordered list) to return directly

    Returns
    -------
    maxima_and_maximisers_sorted :   List of tuples, each tuple being (FOM, optimisation parameters) 
    opt_gratings_sorted          :   List of tuples, each tuple being (FOM, twobox instance)
    chosen_best_grating          :   Twobox instance for the output grating chosen by output_opt_idx
    """

    with open(data_filename, 'rb') as data_file:
        data = pickle.load(data_file)

    opt_FOMs = data["FOM"]
    opt_gratings = data["Optimised grating"]
    opt_params = data["Optimised parameters"] #[0]

    maxima_and_maximisers = zip(opt_FOMs, opt_params)
    maxima_and_gratings = zip(opt_FOMs, opt_gratings)

    # Sort the optima based on the FOM value
    maxima_and_maximisers_sorted = sorted(maxima_and_maximisers, key=itemgetter(0), reverse=True)
    opt_gratings_sorted = sorted(maxima_and_gratings, key=itemgetter(0), reverse=True)
    chosen_best_grating = opt_gratings_sorted[output_opt_idx][1]

    return maxima_and_maximisers_sorted, opt_gratings_sorted, chosen_best_grating