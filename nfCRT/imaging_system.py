# Copyright (c) 2022, Washington University in St. Louis.
#
# All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the Neural Field CRT Dynamic Imaging Library. For more information and source code
# availability see https://github.com/precise-wustl/nf_crt_dynamic_imaging.
#
# Neural Field CRT Dynamic Imaging is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free
# Software Foundation) version 3.0 dated June 2007.

import numpy as np
import torch
class Imaging_system():
    """
    Class that implements the imaging system for the Circular Radon Transform(CRT)
    'views_times' is a dictionary
        - keys represent time steps at which the CRT is calculated
        - entries represent the angles in radians at which CRT is calculated for each time step
    
    Attributes:
    - 'dev':               devices to which tensors are stored
    - 'tR':                radius of transducer arm, field of view is the circle with radius tR
    - 'radii':             radii at which the CRT is calculated
    - 'nQ':                number of quadrature points used for calculating CRT
    - 'views':             views at which the CRT is calculated
    - 'times':             times at which the CRT is calculated
    - 'time_dependence':   Boolean describing the system is dynamic or static
    - 'B':                 sparse tensor with quadrature values, if None will be constructed
    - 'X':                 tensor with quadrature coordinates, if None will be constructed
    
    Methods:
    
    - 'get_quad_info': Calculates the quadrature values and points associated with a given, angle, and time
    - 'generate_imaging_tensors': Forms the tensor 'B' with quadrature values, and the tensor 'X' with quadrature coordinates
    """
    def __init__(self, tR, radii, nQ, views_times, side_length = 1, time_dependence = True, X = None, B = None):
        self.tR = tR
        self.radii = radii
        self.nQ = nQ
        self.side_length = side_length

        self.views = []
        self.times = []
        for time in views_times.keys():
            for view in views_times[time]:
                self.views.append(view)
                self.times.append(time)
        self.time_dependence = time_dependence
        if X is not  None  and B is not None:
            self.X = X
            self.B = B
        else:
            self.generate_imaging_tensors()
    def get_quad_info(self, view, ring, time):
        
        nQr = int(np.ceil(self.nQ*ring))
        phi = np.pi/2 + view + np.pi*np.arange(nQr)/nQr
        offset = ring*np.transpose(np.array([np.cos(phi), np.sin(phi), np.zeros(phi.shape)]))
        quad_points = np.array([ self.tR*np.cos(view), self.tR*np.sin(view), time]) + offset
        
        inds = (np.abs(quad_points[:,0]) <= 1/2)*(np.abs(quad_points[:,1]) <= 1/2)
        quad_points = quad_points[inds,:]
        if not self.time_dependence:
            quad_points = quad_points[:,0:-1]
        quad_vals = quad_points.shape[0]*[np.pi*ring/nQr]

        return quad_points, quad_vals
    
    def generate_imaging_tensors(self):
        nV = len(self.views)
        nR = len(self.radii)

        Bvals = []
        Bind0 = []
        Bind1 = []
        
        if self.time_dependence:
            Xnp = np.zeros( (nV*nR*self.nQ, 3))
        else:
            Xnp = np.zeros( (nV*nR*self.nQ, 2))
        R = 0
        for i, view in enumerate(self.views):
            
            time = self.times[i]
            
            for j, rad in enumerate(self.radii):
                rad = self.radii[j]
                quad_points, quad_vals = self.get_quad_info(view, rad, time)
                r = quad_points.shape[0]
                Xnp[R:R + r, : ] = quad_points
                R += r
                Bvals += quad_vals
                Bind0 += len(quad_vals)*[nR*i + j]
        Xnp = Xnp[0:R,:]
        Bind1 = [ k for k in range(len(Bind0))]
        self.X = torch.from_numpy(Xnp).float()
        Bind =  []
        Bind.append(Bind0)
        Bind.append(Bind1)
        self.B = torch.sparse_coo_tensor(Bind, Bvals, (nR*nV, len(Bind1) ) ).float()