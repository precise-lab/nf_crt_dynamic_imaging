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
import scipy as sc
import scipy.sparse as scs

class DiscreteCircularRadonTransform:
    """
    %SPHERICALTOMO  Creates a 2D spherical Radon tomography test problem
    %
    % This function genetates a tomography test problem based on the spherical
    % Radon tranform where data consists of integrals along circles.  This type
    % of problem arises, e.g., in photoacoustic imaging.
    %
    % The image domain is a square centered at the origin.  The centers for the
    % integration circles are placed on a circle just outside the image domain.
    % For each circle center we integrate along a number of concentric circles
    % with equidistant radii, using the periodic trapezoidal rule.
    %
    % Input:
    %   N           Scalar denoting the number of pixels in each dimesion, such
    %               that the image domain consists of N^2 cells.
    %   angles      Vector containing the angles to the circle centers in
    %               degrees. Default: angles = 0:2:358.
    %   numCircles  Number of concentric integration circles for each center.
    %               Default: numCircles = round(sqrt(2)*N).
    %   asMatrix    If True, a sparse matrix is returned in A (default).
    %               If False, instead a function handle is returned.
    %
    % Output:
    %   A           If input isMatrix is True (default): coefficient matrix with
    %               N^2 columns and length(angles)*p rows.
    %               If input isMatrix is False: A function handle representing a
    %               matrix-free version of A in which the forward and backward
    %               operations are computed as A.fwd(x) and A.bwd(y),
    %               respectively, for column vectors x and y of appropriate size.
    %               The matrix is never formed explicitly, thus saving memory.
    %
    %
    % Based on Matlab code written: Per Christian Hansen, Jakob Sauer Jorgensen, and 
    % Maria Saxild-Hansen, 2010-2017 & Juergen Frikel, OTH Regensburg.
    %
    % This file is part of the AIR Tools II package and is distributed under
    % the 3-Clause BSD License. A separate license file should be provided as
    % part of the package. 
    % 
    % Copyright 2017 Per Christian Hansen, Technical University of Denmark and
    % Jakob Sauer Jorgensen, University of Manchester.
    """
    def __init__(self, N,angles=None,numCircles=None,asMatrix=True):
        self.N = N
        # Default value of the angles to the circle centers.
        if angles is None:
            self.angles = np.arange(0, 359, 2)*np.pi/180.
        else:
            self.angles = angles
            
        if numCircles is None:
            self.numCircles = int( np.round(np.sqrt(2.)*N) )
        else:
            self.numCircles = int( numCircles )
            
        if asMatrix:
            self.A = self._get_or_apply_system_matrix(self.N, self.angles, self.numCircles)
        else:
            self.A = None
            
        self.shape = [self.numCircles*self.angles.shape[0], self.N*self.N]
            
    def fwd(self, x):
        if self.A is not None:
            return self.A*(x)
        else:
            return self._get_or_apply_system_matrix(self.N,self.angles,self.numCircles, x, False)
        
    def bwd(self, y):
        if self.A is not None:
            return self.A.T*y
        else:
            return self._get_or_apply_system_matrix(self.N,self.angles,self.numCircles, y, True)
            
    def _get_or_apply_system_matrix(self, N, angles, numCircles, u=None, adjoint=None):
        # Define the number of angles.
        nA = angles.shape[0]
        
        # Radii for the circles.
        radii  = np.linspace(0,2,numCircles+1)
        radii  = radii[1:]

        # Image coordinates.
        centerImg = np.ceil(N/2)

        # Determine the quarature parameters.
        dx = np.sqrt(2.)/N
        nPhi = np.ceil((4.*np.pi/dx)*radii)
        dPhi = 2*np.pi/nPhi
        
        II = np.arange(nA)
        JJ = np.arange(numCircles)
        
        if u is None:
            isMatrix = True
        else:
            isMatrix = False


        if isMatrix:
            # Initialize vectors that contains the row numbers, the column numbers
            # and the values for creating the matrix A effiecently.
            nnz = int(2*N*nA*numCircles)
            rows = np.zeros( nnz, dtype=np.int32)
            cols = np.zeros( nnz, dtype=np.int32)
            vals = np.zeros( nnz, dtype=np.float64)
            
            idxend = 0

        else:
            if adjoint == False:
                assert u.shape[0] == self.N*self.N
                A = np.zeros(numCircles*nA, dtype=np.float64)
            else:
                assert u.shape[0] == numCircles*nA
                A = np.zeros(self.N*self.N, dtype=np.float64)
    
        # Loop over angles.
        for m in II: 
            # Angular position of source.
            xix = np.cos(angles[m])
            xiy = np.sin(angles[m])
    
            # Loop over the circles.
            for n in JJ:
                # (x,y) coordinates of circle.
                k = np.arange(nPhi[n])*dPhi[n]
                xx = (xix + radii[n]*np.cos(k))/dx + centerImg
                yy = (xiy + radii[n]*np.sin(k))/dx + centerImg
        
                # Round to get pixel index.
                col = np.round( xx )-1
                row = np.round( yy )-1
        
                # Discard if outside box domain.
                IInew = np.logical_and(col>=0, col<N) & np.logical_and(row>=0, row<N)
                row = row[IInew]
                col = col[IInew]
                J = (N-row-1) + col*N
        
                # Convert to linear index and bin
                Ju, w = np.unique(J, return_counts=True)
        
                # Determine rows, columns and weights.
                i = m*numCircles + n
                ii = np.array([i]*Ju.shape[0])
                jj = Ju
                aa = (2.*np.pi*radii[n]/nPhi[n])*w
        
                # Store the values, if any.
                if jj.shape[0] > 0:
                    if isMatrix:
                        # Create the indices to store the values to vector for
                        # later creation of A matrix.
                        idxstart = idxend
                        idxend = idxstart + jj.shape[0]
                        idx = np.arange(idxstart, idxend)
                
                        # Store row numbers, column numbers and values.
                        rows[idx] = ii
                        cols[idx] = jj
                        vals[idx] = aa
                    else:
                        # If any nonzero elements, apply forward or back operator
                        if adjoint==False:
                            A[i] = np.dot(aa, u[jj])
                        else:
                            A[jj] += u[i]*aa
                            
        if isMatrix:
            # Truncate excess zeros.
            rows = rows[:idxend]
            cols = cols[:idxend]
            vals = vals[:idxend]
    
            # Create sparse matrix A from the stored values.
            A = scs.csr_matrix((vals, (rows,cols)), (self.numCircles*nA,self.N*self.N) )/np.sqrt(2)
        return A