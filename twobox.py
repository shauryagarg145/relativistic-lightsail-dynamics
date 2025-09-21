"""
A class to create and simulate a TwoBox grating, storing all grating parameters and hyperparameters.

Inherits from PlotBox and QprBox classes, which provide plotting and 
radiation pressure efficiency calculations respectively.
"""

# IMPORTS ###########################################################################################################################################################################
try:
    from autograd.numpy.numpy_boxes import ArrayBox
except ImportError:
    ArrayBox = None

import grcwa
grcwa.set_backend('autograd')

import torch
import torcwa
# If GPU support TF32 tensor core, the matmul operation is faster than FP32 but with less precision.
# If you need accurate operation, you have to disable the flag below.
torch.backends.cuda.matmul.allow_tf32 = False
sim_dtype = torch.complex128
geo_dtype = torch.float64
#if torch.cuda.is_available():
    #device = torch.device('cuda')
# elif torch.backends.mps.is_available():  # For Apple silicon devices
#     TODO: Requires 32 bit floats. Also leads to RuntimeError: linalg_inv: not supported for complex types yet! in torcwa
#     device = torch.device('mps')
#else:
    #device = torch.device('cpu')
device = torch.device('cpu')

import numpy as np

from autolib import AutoLib
import parameters
I0, L, m, c = parameters.Parameters()
from plotbox import PlotBox
from qprbox import QprBox



class TwoBox(PlotBox, QprBox):
    """
    A TwoBox grating is a grating with two "boxes" (dielectric squares/resonators) in the unit cell. 

    Uses GRCWA or TORCWA library to simulate the grating.
    Simulation is re-run if you change instance variables. 
    All physical lengths pertaining to the grating are in units of the excitation wavelength.

    Attributes
    ----------
    grating_pitch         :   A float for the grating pitch/period 
    grating_depth         :   A float for the grating layer depth/height/thickness 
    box1_width            :   A float for the left box/resonator width
    box2_width            :   A float for the right box/resonator width
    box_centre_dist       :   A float for the distance between the box centres
    box1_eps              :   A float for the left box relative permittivity
    box2_eps              :   A float for the right box relative permittivity
    gaussian_width        :   A float for the Gaussian beam width (metres)
    substrate_depth       :   A float for the substrate layer depth/height/thickness
    substrate_eps         :   A float for the substrate permittiivty
    wavelength            :   A float for the excitation-plane-wave wavelength 
    angle                 :   A float for the excitation-plane-wave angle
    Nx                    :   An integer for the number of grid points in the unit cell
    nG                    :   An integer for the number of Fourier components used in the RCWA simulation
    Qabs                  :   A float for the relaxation parameter, determining the strength of the imaginary frequency and thus smoothness of resonances
    RCWA_engine           :   A string for the RCWA engine to use - 'GRCWA' or 'TORCWA'
    torcwa_edge_sharpness :   An integer for the sharpness of the edge of the unit cell in TORCWA
    title                 :   A string for the title of plots
    fixed_parameters      :   A list for specifying which parameters cannot be changed by setting
                              self.params after initialisation, such as grating_pitch, grating_depth, etc.
    """

    def __init__(self, grating_pitch: float, grating_depth: float, box1_width: float, box2_width: float, box_centre_dist: float, box1_eps: complex, box2_eps: complex, 
                 gaussian_width: float, substrate_depth: float, substrate_eps: float, 
                 wavelength: float=1., angle: float=0.,
                 Nx: float=1000, nG: int=25, Qabs: float=np.inf,
                 RCWA_engine: float='GRCWA', torcwa_edge_sharpness: int=45, fixed_parameters: list=[], 
                 title: str=None,) -> None:

        self.RCWA_engine = RCWA_engine
        
        if self.RCWA_engine == 'GRCWA':
            self.npa = AutoLib('autograd')
        elif self.RCWA_engine == 'TORCWA':
            if Nx < nG*2:
                raise ValueError("Nx must be at least 2*nG for TORCWA")
            if geo_dtype == torch.float64:
                self.npa = AutoLib('torch', device=device, precision='double')
            elif geo_dtype == torch.float32:
                self.npa = AutoLib('torch', device=device, precision='single')
            else:
                raise ValueError("Invalid torch precision. Choose 'double' or 'single'.")
        else:
            raise ValueError("Invalid RCWA engine. Choose 'GRCWA' or 'TORCWA'.")

        self.grating_pitch = self.npa.array(float(grating_pitch))
        self.grating_depth = self.npa.array(float(grating_depth))
        self.box1_width = self.npa.array(float(box1_width))
        self.box2_width = self.npa.array(float(box2_width))
        self.box_centre_dist = self.npa.array(float(box_centre_dist))
        self.box1_eps = self.npa.array(float(box1_eps)) # complex causes problems with FOM (adaptive? gradient? not clear)
        self.box2_eps = self.npa.array(float(box2_eps))  # complex causes problems with FOM (adaptive? gradient? not clear)
        self.gaussian_width = self.npa.array(float(gaussian_width))
        self.substrate_depth = self.npa.array(float(substrate_depth))
        self.substrate_eps = self.npa.array(float(substrate_eps))

        self.fixed_parameters = fixed_parameters
        
        self.wavelength = self.npa.array(float(wavelength))
        self.angle = self.npa.array(float(angle))

        self.Nx = Nx
        self.Ny = 1  # 1D grating simulation, only one grid in the y-direction (transverse to the 1D periodicity)
        self.nG = nG
        self.Qabs = Qabs
        self.torcwa_edge_sharpness = torcwa_edge_sharpness


        self.invert_unit_cell = False

        if title is None:
            self.title = self.RCWA_engine
        else:
            self.title = title

        if self.RCWA_engine == 'GRCWA':
            self.init_RCWA()
        elif self.RCWA_engine == 'TORCWA':            
            if Nx<nG*2:
                raise ValueError("Nx must be at least 2*nG for TORCWA")
            self.init_TORCWA()
        else:
            raise ValueError("Invalid RCWA engine. Choose 'GRCWA' or 'TORCWA'.")

    # Convert the grating parameters to numpy/torch arrays for autodiff compatibility
    # TODO: with GRCWA, it seems we can't float() new parameters before passing to array() 
    #       because ArrayBoxes cannot be float()'d. Is this an issue if other types such as
    #       int are passed through these setters?
    @property
    def grating_pitch(self):
        return self._grating_pitch
    @grating_pitch.setter
    def grating_pitch(self, new_grating_pitch):
        self._grating_pitch = self.npa.array(new_grating_pitch)
    
    @property
    def grating_depth(self):
        return self._grating_depth
    @grating_depth.setter
    def grating_depth(self, new_grating_depth):
        self._grating_depth = self.npa.array(new_grating_depth)
    
    @property
    def box1_width(self):
        return self._box1_width
    @box1_width.setter 
    def box1_width(self, new_box1_width):
        self._box1_width = self.npa.array(new_box1_width)
    
    @property
    def box2_width(self):
        return self._box2_width
    @box2_width.setter
    def box2_width(self, new_box2_width):
        self._box2_width = self.npa.array(new_box2_width)
    
    @property
    def box_centre_dist(self):
        return self._box_centre_dist
    @box_centre_dist.setter
    def box_centre_dist(self, new_box_centre_dist):
        self._box_centre_dist = self.npa.array(new_box_centre_dist)
    
    @property
    def box1_eps(self):
        return self._box1_eps
    @box1_eps.setter
    def box1_eps(self, new_box1_eps):
        self._box1_eps = self.npa.array(new_box1_eps)
    
    @property
    def box2_eps(self):
        return self._box2_eps
    @box2_eps.setter
    def box2_eps(self, new_box2_eps):
        self._box2_eps = self.npa.array(new_box2_eps)
    
    @property
    def gaussian_width(self):
        return self._gaussian_width
    @gaussian_width.setter
    def gaussian_width(self, new_gaussian_width):
        self._gaussian_width = self.npa.array(new_gaussian_width)
    
    @property
    def substrate_depth(self):
        return self._substrate_depth
    @substrate_depth.setter
    def substrate_depth(self, new_substrate_depth):
        self._substrate_depth = self.npa.array(new_substrate_depth)
    
    @property
    def substrate_eps(self):
        return self._substrate_eps
    @substrate_eps.setter
    def substrate_eps(self, new_substrate_eps):
        self._substrate_eps = self.npa.array(new_substrate_eps)
    
    @property
    def wavelength(self):
        return self._wavelength
    @wavelength.setter
    def wavelength(self, new_wavelength): 
        self._wavelength = self.npa.array(new_wavelength)
    
    @property
    def angle(self):
        return self._angle
    @angle.setter
    def angle(self, new_angle):
        self._angle = self.npa.array(new_angle)

    @property
    def all_params(self): 
        self._all_params = [self.grating_pitch, self.grating_depth, 
                            self.box1_width, self.box2_width, self.box_centre_dist, self.box1_eps, self.box2_eps, 
                            self.gaussian_width, self.substrate_depth, self.substrate_eps]
        return self._all_params
    @all_params.setter
    def all_params(self, new_params: list[float]):  # Don't cast new_params to npa.array, else torch gradients will be zero
        self._all_params = new_params
        (self.grating_pitch, self.grating_depth, 
        self.box1_width, self.box2_width, self.box_centre_dist, self.box1_eps, self.box2_eps, 
        self.gaussian_width, self.substrate_depth, self.substrate_eps) = new_params
        self.build_grating_gradable()  # TODO: I think every instance method calls init_RCWA, so this is not needed

    @property
    def params(self):
        """
        List of parameters that can be changed by the user after twobox initialisation. As opposed to 
        self.all_params, the length of self.params varies depending on how many parameters are fixed.

        The order of parameters in self.params is the same as self.all_params, however, without the 
        fixed parameters in self.fixed_parameters.
        
        Manipulate self.params instance variable using getter and setter properties rather than defining
        in __init__. Also, need to define self.params here instead of in __init__. Both of these are
        needed in order for user changes to instance variables to update self.params (and vice versa). 
        """
        self._params = []
        for param_name in parameters.param_names:
            try:
                if param_name not in self.fixed_parameters:
                    self._params.append(getattr(self, param_name))
            except AttributeError:
                print("Warning: self.fixed_parameters was not defined for this grating. Returning all params")
                _all_params = [self.grating_pitch, self.grating_depth, 
                               self.box1_width, self.box2_width, self.box_centre_dist, self.box1_eps, self.box2_eps, 
                               self.gaussian_width, self.substrate_depth, self.substrate_eps]
                return _all_params
        return self._params
    @params.setter
    def params(self, new_params: list[float]):  # Don't cast new_params to npa.array, else torch gradients will be zero
        self._params = new_params
        new_param_idx = 0
        for param_name in parameters.param_names:
            try:
                if param_name not in self.fixed_parameters:
                    setattr(self, param_name, new_params[new_param_idx])
                    new_param_idx += 1
            except AttributeError:
                print("Warning: self.fixed_parameters was not defined for this grating. Ignoring params setter")
        self.build_grating_gradable()  # TODO: I think every instance method calls init_RCWA, so this is not needed


    # Needed for pickling - removes autograd information, written by chatgpt
    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove parts that can't be pickled
        if 'RCWA' in state:
            del state['RCWA']
            del state['npa']
        return self.detach_tensors(state)
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        # TODO: may need to add RCWA/TORCWA init, and redefine npa as these are not pickled.


    def build_grating_gradable(self, sigma: float=100.):
        if self.RCWA_engine == 'GRCWA':
            self.build_grating_GRCWA(sigma)
        elif self.RCWA_engine == 'TORCWA':            
            self.build_grating_torcwa()
        else:
            raise ValueError("Invalid RCWA engine. Choose 'GRCWA' or 'TORCWA'.")
    
    def return_epsilon(self):
        p = self.to_numpy(self.grating_pitch)
        x0 = np.linspace(0, p, self.Nx, endpoint=False)
        if self.RCWA_engine == 'TORCWA':
            self.init_TORCWA()
            # Torcwa does not need flipping this array - check x axis conventions?
            eps_array = self.to_numpy(self.RCWA.return_layer(0,self.Nx,1)[0])
        elif self.RCWA_engine == 'GRCWA':
            self.init_RCWA()
            eps_array = self.RCWA.Return_eps(which_layer=1,Nx=self.Nx,Ny=self.Ny,component='xx')
            # flip to match ordering of desired eps vs grid number - 
            eps_array = np.flip(eps_array)
        return x0,eps_array
            
    def grating_orders(self):
        """Return list of grating orders given current wavelenth and incident angle"""
        # if np.isnan(self.to_numpy(wavelength)): wavelength=self.to_numpy(self.wavelength) 
        # if np.isnan(self.to_numpy(angle)): angle=self.to_numpy(self.angle)
        angle = self.angle
        wavelength = self.wavelength
        p = self.grating_pitch
        
        # Calculate the maximum possible diffraction order
        m_max = self.npa.int((p/wavelength * (1 - self.npa.sin(angle))))
        
        # Iterate over possible diffraction orders from -m_max to m_max
        orders = []
        for m in range(-m_max-1, m_max+1):
            # Calculate sin(θ_m) using the grating equation
            sin_theta_m = (m * wavelength / p) + self.npa.sin(angle)
            # Check if sin(θ_m) is within the valid range [-1, 1]
            if -1 <= sin_theta_m <= 1:
                orders.append(m)
        return orders

        
    def build_grating_GRCWA(self, sigma: float=100.):
        """
        Build the grating permittivity grid as an array of permittivities based on initialised box parameters. 
        
        Since GRCWA is grid-based, continuous changes in box widths or positions must be handled carefully. 
        Here, permittivities are chosen continuously using a softmax probability weighting depending on how 
        far away each grid is from the centre of the boxes. Softmax ensures this array of permittivities is 
        autograd differentiable. A consequence of the softmax is that the boxes are smoother than they should 
        be, with "smoothness" increasing with the temperature parameter 1/sigma.

        Builds box1 as far to the left in the unit cell as possible then fits box2 afterwards. This ensures 
        that large boxes (relative to the grating pitch) can fit inside the unit cell.

        TODO: handle the case where the boxes are too large to fit in the unit cell. Shouldn't necessarily
        throw an error because the optimiser may sometimes step into this region before stepping out.

        Parameters
        sigma :   Softmax inverse temperature, i.e. inverse smoothing factor. Smaller means smoother grating.
        """
        
        Lam = self.grating_pitch
        w1 = self.box1_width
        w2 = self.box2_width
        bcd = self.box_centre_dist
        x1 = w1/2 + 0.02*Lam  # box1 centre location (offset to avoid left box left edge clipping)    
        x2 = x1 + bcd  # box2 centre location    
        eb1 = self.box1_eps
        eb2 = self.box2_eps

        box_separation = bcd - (w1 + w2)/2
        boxes_midpoint = (x1 + w1/2 + x2 - w2/2)/2

        dx = Lam/self.Nx  # grid spacing
        grid_left_boundaries = self.npa.linspace(0,Lam-dx,self.Nx) # does not include x = Lam boundary
        # In this formulation, grid numbers 0, 1, ..., Nx-1 refer to the left boundaries (consistent with x position from 0 to 1*pitch)
        box1_left_boundary = x1-w1/2
        box2_right_boundary = x2+w2/2

        # Build grating by looping across the unit cell grids instead of using index assignment to make build_grating 
        # autograd differentiable.
        grating = [] 
        for grid_left_boundary in grid_left_boundaries:
            # These floats measure how much the current grid fits each condition
            grid_in_box1 = w1/2 - self.npa.abs(grid_left_boundary-x1)
            grid_left_of_box1 = box1_left_boundary - grid_left_boundary
            grid_in_box2 = w2/2 - self.npa.abs(grid_left_boundary-x2)
            grid_between_boxes = box_separation/2 - self.npa.abs(grid_left_boundary - boxes_midpoint)
            grid_right_of_box2 = grid_left_boundary - box2_right_boundary 

            conditions = self.npa.array([grid_in_box1, grid_in_box2, grid_left_of_box1, grid_between_boxes, grid_right_of_box2])
            returns = self.npa.array([eb1, eb2, 1, 1, 1])
            
            probs = self.npa.softmax(conditions,sigma)
            eps = self.npa.sum(probs*returns)

            grating.append(eps)

        if self.invert_unit_cell:
            self.grating_grid = self.npa.array(grating)[::-1]
        else:
            self.grating_grid = self.npa.array(grating)

        return self.npa.array(grating)



    def init_RCWA(self):
        """
        Create GRWCA object for the twobox with the initialised parameters.
        """

        # To simulate a 1D grating rather than 2D PhC, take a small periodicity in the y-direction (L2). 
        # Note: As mentioned in GRCWA documentation (https://github.com/weiliangjinca/grcwa), can only differentiate 
        # wrt photonic crystal period if the ratio of periodicities in the two in-plane directions (x and y) is fixed. 
        # GRCWA encodes this condition by scaling both (reciprocal) lattice vectors after they've been created in the 
        # kbloch.py module. Hence, set unity grating vector here and use Pscale kwarg in Init_Setup() to scale the period 
        # accordingly.

        dy = 1e-4 
        L1 = [1.,0]
        L2 = [0,dy] 

        freq = 1/self.wavelength  # frequency is 1/wavelength when c = 1
        freqcmp = freq*(1+1j/2/self.Qabs)

        theta = self.angle # radians
        phi = 0.

        # setup RCWA
        obj = grcwa.obj(self.nG,L1,L2,freqcmp,theta,phi,verbose=0) # verbose=1 for debugging, prints ng actually used 

        # add layers

        eps_vacuum = 1
        vacuum_depth = self.wavelength

        obj.Add_LayerUniform(vacuum_depth,eps_vacuum)
        obj.Add_LayerGrid(self.grating_depth,self.Nx,self.Ny)
        if self.substrate_eps != 0:
            obj.Add_LayerUniform(self.substrate_depth,self.substrate_eps)
        obj.Add_LayerUniform(vacuum_depth,eps_vacuum)
        obj.Init_Setup(Pscale=self.grating_pitch)

        # TODO: re-building the grating every time we calculate diffraction efficiencies is inefficient 
        #       because changes to parameters such as wavelength do not change the grating parameters.
        self.build_grating_gradable()  # update twobox whenever user changes box parameters
        obj.GridLayer_geteps(self.grating_grid)


        planewave = {'p_amp':0,'s_amp':1,'p_phase':0,'s_phase':0}
        obj.MakeExcitationPlanewave(planewave['p_amp'],planewave['p_phase'],planewave['s_amp'],planewave['s_phase'],order = 0)

        self.RCWA = obj
        return obj


    def eff(self):
        """
        Calculates -1 <= m <= 1 reflection/transmission efficiencies for the twobox.
        """
        if self.RCWA_engine == 'GRCWA':
            self.init_RCWA()
            R_byorder,T_byorder = self.RCWA.RT_Solve(normalize=1, byorder=1)
            Fourier_orders = self.RCWA.G

            Rs = []
            Ts = []
            RT_orders = [-1,0,1]
            # IMPORTANT: have to use append method to a list rather than index assignment
            # Else, autograd will throw a TypeError with float() argument being an ArrayBox
            for order in RT_orders:
                Rs.append(self.npa.sum(R_byorder[Fourier_orders[:,0]==order]))
                Ts.append(self.npa.sum(T_byorder[Fourier_orders[:,0]==order]))
        elif self.RCWA_engine == 'TORCWA':
            self.init_TORCWA()
            RT_orders = self.grating_orders()
            if len(RT_orders) > 3:
                raise NotImplementedError("More than 3 orders detected, not currently supported for FOM calculation. Check grating pitch to wavelength ratio.")
            
            Rs = self.npa.zeros(len([-1,0,1]))
            Ts = self.npa.zeros(len([-1,0,1]))
            orders = [[j,0] for j in RT_orders]
            
            lRs = self.npa.abs(self.npa.power(self.RCWA.S_parameters(orders=orders, direction='forward', port='reflection', polarization='yy', ref_order=[0,0], power_norm=True),2))
            lTs = self.npa.abs(self.npa.power(self.RCWA.S_parameters(orders=orders, direction='forward', port='transmission', polarization='yy', ref_order=[0,0], power_norm=True),2))
            for i,j in enumerate(RT_orders):
                Rs[1+j] = lRs[i]
                Ts[1+j] = lTs[i]
        return Rs,Ts


    # TORCWA methods
    def init_TORCWA(self):
        """
        Initialise the TORCWA solver
        """

        # Empty GPU cache to avoid memory issues
        # torch.cuda.empty_cache()
        # Grating
        # To simulate a 1D grating, take a small periodicity in the y-direction. 
        # The grating is in the x-direction.
        dy = 1e-4 
        L1 = [1.,0]
        L2 = [0,dy] 
        
        # testing if this helps with jacobian
        # self.wavelength=self.npa.array(self.wavelength) #,dtype=geo_dtype,device=device)
        # end test
        freq = self.npa.array(1/self.wavelength, dtype=geo_dtype, device=device) # freq = 1/wavelength when c = 1
        freqcmp = freq*(1+1j/2/self.Qabs)

        # Incoming wave
        theta = self.angle # radians
        phi = 0.

        # Setup TORCWA
        # geometry
        L = [self.grating_pitch, dy]            # nm / nm size of unit cell
        torcwa.rcwa_geo.dtype = geo_dtype
        torcwa.rcwa_geo.device = device
        torcwa.rcwa_geo.Lx = L[0]
        torcwa.rcwa_geo.Ly = L[1]
        torcwa.rcwa_geo.nx = self.Nx
        torcwa.rcwa_geo.ny = 2 # np.min(self.Ny,2) # 2 minimum for 2d simulation displaying ? 
        torcwa.rcwa_geo.grid()
        torcwa.rcwa_geo.edge_sharpness = self.torcwa_edge_sharpness
        sim = torcwa.rcwa(freq=freq, order=[self.nG,0], L=L, dtype=sim_dtype, device=device, stable_eig_grad=False) 
        # 4/3/25 added stable_eig_grad=False to debug jacobian not working. Without this flag, 
        # self.Eig doesn't work, but with it, grad sometmies returns ill defined eigenvector error 
        # when calling grad, instead of NaN - both it seems only for orders past cutoff (tbc)
        
        eps_vacuum = 1        
        sim.add_input_layer(eps=eps_vacuum)  # input and output layers are eps=mu=1 by default, so this line not needed
        sim.set_incident_angle(inc_ang=theta, azi_ang=phi)  # for some reason throws an error in solve_global_smatrix if this line is before defining input layer   
        self.build_grating_torcwa()
        sim.add_layer(thickness=self.grating_depth, eps=self.grating_grid_torcwa)
        sim.add_layer(thickness=self.substrate_depth, eps=self.substrate_eps)
        sim.solve_global_smatrix()
        self.RCWA = sim
        
    def build_grating_torcwa(self):
        """
        Build the grating for the TORCWA solver using the twobox parameters
        no care taken for autograd, assuming torcwa/torch will handle this
        """
        
        dy = 1e-4
        Lam = self.grating_pitch
        L = [Lam, dy]
        w1 = self.box1_width
        w2 = self.box2_width
        eb1 = self.box1_eps
        eb2 = self.box2_eps
        if self.invert_unit_cell:
            w1, w2 = w2, w1
            eb1, eb2 = eb2, eb1
        bcd = self.box_centre_dist
        x1 = w1/2 + 0.02*Lam # box1 centre location (offset to avoid left box left edge clipping)
        x2 = x1 + bcd # box2 centre location    
        
        box1_bool = torcwa.rcwa_geo.rectangle(Wx=w1, Wy=L[1], Cx=x1, Cy=L[1]/2.)  # width, height, centerx, centery
        box2_bool = torcwa.rcwa_geo.rectangle(Wx=w2, Wy=L[1], Cx=x2, Cy=L[1]/2.) # width, height, centerx, centery
        layer0_bool = torcwa.rcwa_geo.union(box1_bool,box2_bool)
        layer0_eps = eb1*box1_bool + eb2*box2_bool + (1. - layer0_bool)
        self.grating_grid_torcwa = layer0_eps  #: TODO: why not cast to grating_grid directly?
        
        try: # when called to calculate gradient functions rather than values, tensors are virtual - do not copy to grating_grid
            self.grating_grid = self.to_numpy(layer0_eps)
        except:  # TODO: catch specific error
            self.grating_grid = np.zeros((self.Nx,0))
        return self.grating_grid


    def to_numpy(self,x):
        """ 
        Converts tensors, autograd arrays or numpy arrays, or list or tuples of these 
        (including mixed tuples) to numpy arrays (or tuples of these). For scalars, 
        output are native python, not numpy, for easier readability in print statements.
        All results are separated from gradient information.
        """        
        if isinstance(x,(list,tuple)):        
            result = []
            for item in x:
                if isinstance(item, torch.Tensor):
                    # Convert tensor to numpy array.
                    if item.numel()==1:
                        result.append(item.item())
                    else:
                        result.append(item.detach().cpu().numpy())
                elif(isinstance(item, (tuple,list))):
                    # nested tuple -> recurse
                    result.append(self.to_numpy(item)) # recurse for nested tuples
                elif ArrayBox is not None and isinstance(x, ArrayBox):
                    # autograd array
                    if item.size==1:
                        result.append(np.asarray(x).item())
                    else:
                        result.append(np.array(item))
                elif isinstance(item, (np.ndarray)):
                    result.append(np.array(item))
                elif np.isscalar(item):
                    result.append(item)
                else:
                    raise TypeError(f"to_numpy Unsupported type: {type(item)}")
            
            if isinstance(x,tuple):
                if len(result)==1:
                    return result[0]
                else:
                    return tuple(result)        
            
            if isinstance(x,list):
                try:
                    return np.array(result)
                except:
                    return result
            else:
                return result
        else:        
            if(isinstance(x, torch.Tensor)):
                return x.detach().cpu().numpy()
            else:
                return np.array(x)

    def detach_tensors(self,obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach()
        elif isinstance(obj, list):
            return [self.detach_tensors(x) for x in obj]
        elif isinstance(obj, tuple):
            return tuple(self.detach_tensors(x) for x in obj)
        elif isinstance(obj, dict):
            return {k: self.detach_tensors(v) for k, v in obj.items()}
        elif hasattr(obj, '__dict__'):
            # If the object is a custom class instance, create a shallow copy
            # and recursively detach tensors in its __dict__
            new_obj = obj.__class__.__new__(obj.__class__)
            new_obj.__dict__ = self.detach_tensors(obj.__dict__)
            return new_obj
        else:
            return obj