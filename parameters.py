"""
A module to store parameters of a bigrating + laser configuration, and optimisation parameters.
Some functions are included to calculate velocity-dependent terms from special relativity, including 
the Lorentz gamma factor and Doppler factor.

TODO: move relativity functions to specrel.py
"""

import numpy as np
import scipy


def gamma_ND(v):
    """
    Calculate the Lorentz gamma factor with an input speed/velocity normalised by the speed of light.

    Parameters
    ----------
    v :   Speed (float), two/three-velocity (list or np array), or list of two/three-velocities

    Returns
    -------
    gamma :   Lorentz gamma factor
    """
    if not isinstance(v,(list,np.ndarray)):
        v = [v]
    v = np.array(v)
    
    if any(isinstance(i, np.ndarray) for i in v):
        vnorm = np.linalg.norm(v,axis=1)
    else:
        vnorm = np.linalg.norm(v)
    
    gamma = 1/np.sqrt(1-np.power(vnorm,2))
    return gamma

def D1_ND(v):
    """
    Calculate the Doppler factor with an input velocity normalised by the speed of light.

    Parameters
    ----------
    v :   Speed (float), two/three-velocity (list or np array), or list of two/three-velocities

    Returns
    -------
    D1 : Doppler factor
    """
    if not isinstance(v,(list,np.ndarray)):
        v = [v]
    v = np.array(v)
    
    if any(isinstance(i, np.ndarray) for i in v):
        vx = np.array([i[0] for i in v])
    else:
        vx = np.array(v[0])

    D1 = gamma_ND(v)*(1-vx)
    return D1



L = 10  # grating width (metres in 2D model)
def Parameters():
    P = 50e9  # laser power (watts)
    I0 = P/L  # laser power per unit grating length
    m = 1/1000  # mass (kilograms)
    c = scipy.constants.c
    return I0, L, m, c


wavelength = 1.  # Laser wavelength
final_speed = 20.  # percentage of c
fixed_pitch = 1.227 # If the pitch is fixed, other parameters like box widths are naturally constrained by this value
param_names = ["grating_pitch", "grating_depth", 
                "box1_width", "box2_width", "box_centre_dist", 
                "box1_eps", "box2_eps", 
                "gaussian_width", "substrate_depth", "substrate_eps"]  # Names of all optimisable twobox parameters
fixed_parameters = ["gaussian_width"]  # Fix parameters during optimisation
fix_parameter_values = [2*L]  # Values of fixed parameters, in the same order as fixed_parameters
def Hyperparameters():
    # Engine parameters
    RCWA_engine = "TORCWA"
    torcwa_sharpness = 45

    angle = 0.
    Nx = 100  # Number of grid points for RCWA simulation

    # Number of Fourier components for RCWA simulation
    if RCWA_engine == "TORCWA":
        nG = 12
    elif RCWA_engine == "GRCWA":
        nG = 25
    else:
        raise ValueError("RCWA engine not recognised. Please use 'TORCWA' or 'GRCWA'.")

    # Relaxation parameter should be np.inf unless you need to avoid singular matrix at grating cutoffs
    # Note that the optimiser will only find (likely unphysical) large-magnitude, noisy rNeg1 optima when Qabs = np.inf 
    Qabs = 1e7
    goal = 0.1  # Stopping criteria for adaptive sampling in the FOM (set float for loss_goal, set int for npoints_goal)
    return_grad = True  # Return FOM and gradient of FOM

    return wavelength, angle, Nx, nG, Qabs, goal, final_speed, return_grad, RCWA_engine, torcwa_sharpness, fixed_parameters


choose_monofom = "asymp"
#choose_multifom = "uniform"
choose_multifom = "monochrome"
def FOMSettings():
    # See fom.py for FOM options and kwargs  
    fom_kwargs = {"use_perturbed": False}
    return choose_monofom, choose_multifom, fom_kwargs


def OptimisationSettings():
    # Global optimisation parameters
    num_cores = 2  # number of cores to run parallel optimisation
    maxtime = 2  # Stop after maxtime minutes
    maxstop = {'maxtime': maxtime}  # global 1000
    if choose_multifom != "monochrome":
        runID = f"F{choose_monofom}{int(final_speed)}_fixgaussian20_50GW"  # ID for saving results to distinguish different runs
    else:
        runID = f"F{choose_monofom}{choose_multifom}_fixgaussian20_50GW"  # ID for saving results to distinguish different runs

    # Local optimisation parameters
    xtol_rel = 1e-4  
    ftol_rel = 1e-8  

    seed = 20250714  # LDS seed
    sampling = 'sobol'  # 'sobol' or 'random'
    n_sample_exp = 4
    n_sample = 2**n_sample_exp  # number of random samples per iteration, the best of which (in non-overlapping regions of attraction) are locally optimised

    return num_cores, maxtime, maxstop, runID, xtol_rel, ftol_rel, seed, sampling, n_sample_exp, n_sample


mirror_substrate_depth = 1.  # Depth of the substrate if mirror_substrate is true (wavelength units)
mirror_substrate_eps = -1e6  # Permittivity of the substrate if mirror_substrate is true
def Bounds():
    ## Parameter bounds
    # Pitch bounds have been set to avoid ±1 or ±2 grating cutoffs, because the grating is rotating.
    # The minimum pitch must be set because any smaller pitches would result in the +1 order being cutoff for small rotation angles. 
    # The maximum pitch must be set because any larger pitches would result in the -2 order appearing for small rotation angles. 
    # The +1 and -2 orders are selected because they appear/disappear before the -1/+2 orders (at positive rotation angle)
    # wavelength_max = wavelength/D1_ND(final_speed/100)
    wavelength_max = 1.
    max_angle_cutoff1 = 0.1*np.pi/180  # maximum angle before order +1 is evanescent
    min_angle_cutoff2 = 15*np.pi/180  # minimum angle before order -2 is non-evanescent
    # pitch_min = np.round(1*wavelength_max/(1 - np.sin(max_angle_cutoff1)), 3)  
    # pitch_max = np.round(2*wavelength_max/(1 + np.sin(min_angle_cutoff2)), 3)

    pitch_min = np.round(1*wavelength_max/(1 - np.sin(0.01*np.pi/180)), 3)  
    pitch_max = np.round(1*wavelength_max/(1 - np.sin(0.1*np.pi/180)), 3)  

    h1_min = 0.01*fixed_pitch  # Offset from zero to avoid zero Jacobian determinant 
    h1_max = 1.5*fixed_pitch

    box_width_min = 0.01*fixed_pitch  # Offset from zero to avoid zero Jacobian determinant
    box_width_max = 1.*fixed_pitch  # single box width must be smaller than pitch

    box_centre_dist_min = 0.03*fixed_pitch  # Offset from zero to avoid zero Jacobian determinant and symmetric unit cell
    box_centre_dist_max = 0.5*fixed_pitch  # redundant space if > 0.5*pitch

    box_eps_min = 1.1**2  # Minimum allowed grating permittivity set above vacuum to avoid zero Jacobian determinant 
    box_eps_max = 3.5**2  # Maximum allowed grating permittivity set to silicon

    gaussian_width_min = 0.1*L 
    gaussian_width_max = 10*L

    substrate_depth_min = h1_min  # Offset from zero to avoid zero Jacobian determinant 
    substrate_depth_max = 1.5*fixed_pitch 
    substrate_eps_min = box_eps_min 
    substrate_eps_max = box_eps_max

    # # All params
    # param_bounds = [(pitch_min, pitch_max), (h1_min, h1_max), 
    #                 (box_width_min, box_width_max), (box_width_min, box_width_max),
    #                 (box_centre_dist_min, box_centre_dist_max),
    #                 (box_eps_min, box_eps_max), (box_eps_min, box_eps_max),
    #                 (gaussian_width_min, gaussian_width_max),                    
    #                 (substrate_depth_min, substrate_depth_max),
    #                 (substrate_eps_min, substrate_eps_max)]
    
    # # Fixed pitch
    # param_bounds = [(h1_min, h1_max),
    #                 (box_width_min, box_width_max), (box_width_min, box_width_max),
    #                 (box_centre_dist_min, box_centre_dist_max),
    #                 (box_eps_min, box_eps_max), (box_eps_min, box_eps_max),
    #                 (gaussian_width_min, gaussian_width_max),
    #                 (substrate_depth_min, substrate_depth_max),
    #                 (substrate_eps_min, substrate_eps_max)]

    # # Fixed pitch and gaussian
    # param_bounds = [(h1_min, h1_max),
    #                 (box_width_min, box_width_max), (box_width_min, box_width_max),
    #                 (box_centre_dist_min, box_centre_dist_max),
    #                 (box_eps_min, box_eps_max), (box_eps_min, box_eps_max),
    #                 (substrate_depth_min, substrate_depth_max),
    #                 (substrate_eps_min, substrate_eps_max)]
    
    # Fixed gaussian
    param_bounds = [(pitch_min, pitch_max), (h1_min, h1_max),
                    (box_width_min, box_width_max), (box_width_min, box_width_max),
                    (box_centre_dist_min, box_centre_dist_max),
                    (box_eps_min, box_eps_max), (box_eps_min, box_eps_max),
                    (substrate_depth_min, substrate_depth_max),
                    (substrate_eps_min, substrate_eps_max)]

    # # Fixed substrate and pitch
    # param_bounds = [(h1_min, h1_max), 
    #                 (box_width_min, box_width_max), (box_width_min, box_width_max),
    #                 (box_centre_dist_min, box_centre_dist_max),
    #                 (box_eps_min, box_eps_max), (box_eps_min, box_eps_max),
    #                 (gaussian_width_min, gaussian_width_max)]
    
    return h1_min, h1_max, param_bounds
