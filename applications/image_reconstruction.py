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

import torch
import numpy as np
import scipy.io as io

import argparse
import sys
sys.path.append( "../")
import os
from nfCRT import *

    


def data_fidelity(A, nf, X, measurements, std):
    """Computes data fidelity term for image reconstruction problem

    Args:
        A (pytorch sparse matrix): circular radon transform matrix
        nf (NeuralField): neural field object 
        X (pytorch tensor): quadrature point for circular radon transform
        measurements (pytorch tensor): measurements used for reconstruction
        std (double): noise standard deviation

    Returns:
        tensor: data fidelity 
    """
    OUT = nf(X)
    exp_measurements = A@OUT
    sq = (exp_measurements - measurements)**2
    loss = torch.mean(sq)/(2*std**2) 
    return loss

def nf_loss(A, nf, X, \
                    measurements, std, \
                    delta, gamma, 
                    rho):
    """

    Args:
        A (pytorch sparse matrix): circular radon transform matrix
        nf (NeuralField): neural field object 
        X (pytorch tensor): quadrature point for circular radon transform
        measurements (pytorch tensor): measurements used for reconstruction
        std (double): noise standard deviation
        delta (double):  total variation regularization weight
        gamma (double): l_{1/2} training regularization weight on partition
        rho (_type_): Frobenius training regularization weight on coefficients

    Returns:
        loss: sum of data fidelity and regularization
    """

    

    loss = data_fidelity(A, nf, X, measurements, std)
    if delta > 0 or gamma > 0:
        gradx = torch.rand(10**5, nf.psi.geodim, device = X.get_device(), requires_grad = True) - 1/2
        grad_psi = nf.psi(gradx)
        if delta > 0:
            out = torch.einsum( 'ij,jk,ki -> i', grad_psi, nf.C, nf.P(gradx))
            J = torch.autograd.grad(out, gradx, grad_outputs = torch.ones_like(out), create_graph = True)[0]
            loss += delta*torch.mean(torch.norm(J,dim = 1))
        if gamma > 0:
            loss += gamma*torch.mean(torch.sum((grad_psi + 1e-2)**(1/2),1)**2)

    if rho > 0:
        loss += rho*torch.mean(nf.C**2)
    return loss

def create_measurements(vpt, nA, rnl):
    """Runs forward model from ground truth data and creates synthetic measurements

    Args:
        vpt (int): number of views per frame
        nA (int): number of time steps
        rnl (double): relative noise level

    Returns:
        (numpy array, numpy array, numpy array, double): array of measurements, time average measurements, radii, and noise standard deviation
    """

    image = io.loadmat('induced_pressure')['induced_pressure'] 
    nR = int(np.sqrt(2)*image.shape[0]/2) + 1
    radii = np.linspace(0, np.sqrt(2) , nR + 1)
    radii = radii[1:]
    
    a = DiscreteCircularRadonTransform(image.shape[0], numCircles= nR)
    image = image.flatten()
    
    time_int = np.linspace(-1/2, 1/2,nA)
    

    #Calculates measurements and average measurementsf
    ave_meas = np.zeros(nR*nA)
    angles_per_time_vals = {}
    angles_per_time_inds = {}
    print("         Calculating views and times for average")
    for i in range(nA):
        angles_per_time_vals[time_int[i]] = []
        angles_per_time_inds[i] = []
        for k in range(vpt):
            angles_per_time_vals[time_int[i]].append(2*np.pi*( i/nA + k/vpt) + np.pi/2 )
            angles_per_time_inds[i].append(  int( (i + (nA/vpt)*k)%nA))

    measurements = np.zeros(nR*nA*vpt) 
    print("         Calculating Measurements")
    i = 0
    for time_index in angles_per_time_inds.keys():
        holder = a.fwd(image[time_index::nA])
        for view_index in angles_per_time_inds[time_index]:
            measurements[i*nR:(i+1)*nR] = holder[view_index*nR:(view_index+1)*nR]
            i += 1
    
    std = rnl*np.max(measurements)
    
    print('Noise Standard Deviation = ', std)
    measurements += np.random.normal(0, std, nR*nA*vpt)
    i = 0
    for time_index in angles_per_time_inds.keys():
        for view_index in angles_per_time_inds[time_index]:
            ave_meas[view_index*nR:(view_index+1)*nR] += measurements[i*nR:(i+1)*nR]/vpt
            i += 1
    return measurements, ave_meas, radii, std

def form_static_systems(ave_meas, rads, tR, nB, NQ):
    """Forms static systems for solving time averaged reconstruction problem

    Args:
        ave_meas (numpy array): time averaged measurements
        rads (numpy array): list of radii in circular Radon transform
        tR (double): transducer arc radius
        nB (int): number of batches
        NQ (int): number of quadrature points per unit arc
        dev (pytorch device): device for pytorch tensors to be located

    Returns:
        (list, list): list of static systems, list of static measurements
    """
    NR = len(rads)
    NA = int(len(ave_meas)/(NR))
    ave_dict = {}
    ave_dict[0] = [2*np.pi*( i/NA ) + np.pi/2  for i in range(NA)]
    MEAS = np.reshape(ave_meas, (NA, NR))
    ave_systems = []
    static_targets = []
    for j in range(nB):
        measj = MEAS[:, j::nB].flatten()
        static_targets.append(torch.from_numpy(measj).float())
        ave_systems.append(Imaging_system(tR, radii[j::nB], NQ, ave_dict, time_dependence = False))
    return ave_systems, static_targets

def form_dynamic_systems(nA, rads, tR, vpt, ntB, nrB, NQ):
    """Forms dynamic systems for solving dynamic reconstruction problem

    Args:
        nA (int): number of time frames
        rads (numpy array): list of radii in circular Radon transform
        tR (double): transducer arc radius
        vpt (int): views per frame
        ntB (int): number of bathes across time steps
        nrB (int): number of bathes across radii
        NQ (int): number of quadrature points per unit arc
        dev (pytorch device): device for pytorch tensors to be located

    Returns:
        list: list of dynamic imaging systems
    """
    dynamic_systems = []
    time_int = np.linspace(-1/2, 1/2,nA)
    
    for j in range(ntB):
        for l in range(nrB):
            #Gets randomly chosen points for Gradient evaluation and regularization
            print("     Forming dynamic system", j*nrB+l)
            angles_per_time_vals = {}
            i = j
            print("         Calculating views and times")
            while i < nA:
                angles_per_time_vals[time_int[i]] = []
                for k in range(vpt):
                    angles_per_time_vals[time_int[i]].append(2*np.pi*( i/nA + k/vpt) + np.pi/2 )
                i += ntB
            print("     Forming Tensors, this may take a bit")
            dynamic_systems.append(Imaging_system(tR, rads[l::nrB], NQ, angles_per_time_vals))
    return dynamic_systems

def get_dynamic_targets(measurements, rads, TB, RB, NA, vpt):
    """Forms dynamic targets

    Args:
        measurements (numpy array): measurements
        rads (numpy array): radii used in circular radon transform
        TB (int): number of batches in time
        RB (int): number of bathes across radii
        NA (int): number of time frames
        vpt (int): views per frame

    Returns:
        list: dynamic targets
    """
    dynamic_targets = []
    for j in range(TB):
        for l in range(RB):
            #Gets randomly chosen points for Gradient evaluation and regularization
            angles_per_time_inds = {}
            i = j
            while i < NA:
                angles_per_time_inds[i] = []
                for k in range(vpt):
                    angles_per_time_inds[i].append(  int( (i + (NA/vpt)*k)%NA))
                i += TB
            
            nr = len(rads[l::RB])
            na = int((i -j)/TB )*vpt
            meas = np.zeros( nr*na)

            i = 0
            for time_index in angles_per_time_inds.keys():
                for k in range(vpt):
                    meas_ind = time_index*vpt*nR + k*nR
                    meas[i*nr: (i+1)*nr] = measurements[meas_ind +l: meas_ind + nR:RB]
                    i += 1
            dynamic_targets.append(torch.from_numpy(meas).detach().float().to(dev))
    return dynamic_targets

def train_network(nf, systems, targets,\
            delta, std, \
            nI, inner_thresholdC, inner_threshold_pou, dev, condition_weights = False):
    """Trains neural field

    Args:
        nf (NeuralField): neural field
        systems (list of Imaging_system): imaging matrices and quadrature coordinates
        targets (list of pytorch tensors): target measurements for training
        std (double): noise standard deviation
        inner_thresholdC (double): cutoff threshold for coefficient gradient
        inner_threshold_pou (double): cutoff threshold for pou gradient
        condition_weights (bool, optional): Determines to utilize training regularization. Defaults to False.
    """
    

    if  condition_weights:
        gamma = 1
        rho = 1e-4/std**2
        Lrmax = 1e-3
        Lrmin = 1e-6
    else: 
        gamma = 0
        rho = 0
        Lrmax = 1e-4
        Lrmin = 1e-7
    Lr_shinkage = (1e-3)**(2/nI)

    Lr_phi = Lrmax
    LrC = Lrmax

    weight_shrinkage = (0.1)**(8/nI)
    #Number of inner iterations
    
    ni = min([20, int(np.ceil(320/len(targets)))])

    for i in range(nI + 1):
        if i > 0:
            acum_loss0 = acum_loss
        if i < nI/4:
            rho *= weight_shrinkage
            gamma *= weight_shrinkage
        else:
            rho = 0
            gamma = 0
        print("     Outer Iteration", i)
        
        if delta > 0:
            optimizer0 = torch.optim.Adam([nf.C], lr = LrC)
            optimizer1 = torch.optim.Adam(nf.psi.parameters(), lr = Lr_phi)
        else:
            optimizer0 = torch.optim.Adam([nf.C], lr = LrC, betas = (0.8, 0.9))
            optimizer1 = torch.optim.Adam(nf.psi.parameters(), lr = Lr_phi, betas = (0.8, 0.9))
        
        for k in range(ni+1):
            gradC = 0
            acum_loss = 0

            for system, target in zip(systems, targets):
                torch.cuda.empty_cache()
                optimizer0.zero_grad()
                loss = nf_loss(system.B.to(dev), nf, system.X.to(dev),  \
                            target.to(dev), std, \
                            delta, gamma, rho)
                acum_loss += loss.detach()
                loss.backward()
                optimizer0.step()
                gradC += torch.sum(nf.C.grad**2)
            acum_loss /= len(targets)
            if k == 0:
                gradC0 = gradC
            elif gradC/gradC0 < inner_thresholdC**2:
                print(  "   Threshold reached Coeff")
                break
            if k == ni:
                LrC *= Lr_shinkage
                print("     Shrinking LrC")
        
        print("     Loss after Updating Coefficients = ", acum_loss)
        for k in range(ni + 1):
            grad_pou = 0
            acum_loss = 0

            for system, target in zip(systems, targets):
                torch.cuda.empty_cache()
                optimizer1.zero_grad()
                loss = nf_loss(system.B.to(dev), nf, system.X.to(dev),  \
                            target.to(dev), std, \
                            delta, gamma, rho)
                acum_loss += loss.detach()
                loss.backward()
                optimizer1.step()
                for param in nf.psi.parameters():
                    grad_pou += torch.sum(param.grad**2)
            acum_loss /= len(targets)
             
            if k == 0:
                grad_pou0 = grad_pou
            elif grad_pou/grad_pou0 < inner_threshold_pou**2:
                print("     Threshold reached Network")
                break
            if k == ni:
                Lr_phi *= Lr_shinkage
                print("     Shrinking Lr")
        print("     Loss after Training Network = ", acum_loss)
        if Lr_phi < Lrmin  or LrC < Lrmin:
            print("     Learning Rates are too small")
            break
        if i > 0:
            if acum_loss > (0.995)*acum_loss0: #(0.999)*acum_loss0:
                break

def spatial_to_dynamic( dnf, snf):
    """Initializes dynamic neural field to match static neural field

    Args:
        dnf (NeuralField): dynamic neural field
        snf (NeuralField): static neural field
    """
    print("-"*15, "Initializing Dynamic Network", "-"*15)

    nI = 2
    ni = 100
    Lr = 1e-3
    optimizer1 = torch.optim.Adam(dnf.psi.parameters(), lr = Lr)
    for i in range(nI*ni):
        x = torch.rand(10**6, dnf.psi.geodim, device = dnf.C.get_device()) - 1/2
        psi = dnf.psi(x)
        sphi = snf.psi(x[:,0:-1])
        optimizer1.zero_grad()
        loss = torch.mean( (psi[:,0:snf.psi.N_part] - sphi)**2)
        loss.backward()
        optimizer1.step()

    Lr = 1e-2
    optimizer0 = torch.optim.Adam([dnf.C], lr = Lr)
    for i in range(ni):
        x = torch.rand(10**6, dnf.psi.geodim, device = dnf.C.get_device()) - 1/2
        loss = torch.mean((dnf(x) - snf(x[:,0:-1]))**2) 
        optimizer0.zero_grad()
        loss.backward()
        optimizer0.step()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Matrix Free Implementation of CRT.')
    parser.add_argument('--out_folder', #Folder where neural field is saved
                        default=  'VPT4/',
                        type = str)
    parser.add_argument('--vpt', #Number of views per time
                        default = 4,
                        type = int)
    parser.add_argument('--rnl', #Relative noise level
                        default = 0.025,
                        type = float)
    parser.add_argument('--nT', #Number of time steps 
                        default = 180,
                        type = int)
    parser.add_argument('--nQuad_points', #Number of quadrature points per unit arc for CRT
                        default = 300,
                        type = int)
    parser.add_argument('--naB', #Number of batches for time averaged reconstruction problem
                        default = 2,
                        type = int)
    parser.add_argument('--ntB', #Number of batches across time steps
                        default = 2,
                        type = int)
    parser.add_argument('--nrB', #Number of batches across radii
                        default = 4,
                        type = int)
    parser.add_argument('--nI', #Maximum number of iterations for training neural field
                        default = 40,
                        type = int)
    args = parser.parse_args()

    print('#'*20, args.vpt, " Views per time and ", args.rnl, " PSNR", '#'*20)
    if torch.cuda.is_available():
        print("Cuda available, Implementing with GPU")
        dev = torch.device('cuda:0')
    else:
        print("Cuda not available, Implementing with CPU")
        dev = torch.device('cpu')
    try:    
        os.mkdir(args.out_folder)
    except:
        pass
    
    measurements, ave_meas, radii, std = create_measurements(args.vpt, args.nT, args.rnl)
    nR = len(radii)
    print("Number of circles = ", nR)
    tR = np.sqrt(2)/2

    ave_systems, static_targets = form_static_systems(ave_meas, radii, tR, args.naB,  args.nQuad_points)
    dynamic_systems = form_dynamic_systems(args.nT, radii, tR, args.vpt, args.ntB, args.nrB, args.nQuad_points)
    dynamic_nB = args.ntB*args.nrB

    #Geometric Dimension
    geodim = 3

    #POUnet
    Width_static = 100
    Width_dynamic = 140
    number_of_partions_static = 20
    number_of_partions_dynamic = 40
    Depth = 4
    static_pou = POU_Siren(Width_static, number_of_partions_static, geodim = geodim - 1, Depth = Depth).to(dev)
    dynamic_pou = POU_Siren(Width_dynamic, number_of_partions_dynamic, geodim = geodim, Depth = Depth).to(dev)
    #Polynomial Basis
    mx = 3
    mt = 3
    #Polynomial Basis
    static_P = Polynomial(geodim - 1, mx, st_separability = False )
    dynamic_P = Polynomial(geodim, mx, mt = mt, st_separability = True )
    
    #Spanning Coefficients
    static_neural_field = NeuralField(static_pou, static_P, dev)
    dynamic_neural_field = NeuralField(dynamic_pou, dynamic_P, dev)
    #POUnet and C training
    sep = '-'*15
    inner_thresholdC = 1e-2
    inner_threshold_pou = 1e-2

    #Static optimization
    print(sep, "Static Component", sep)
    
    train_network(static_neural_field, ave_systems, static_targets,\
                    0., std, args.nI, inner_thresholdC, inner_threshold_pou, dev, condition_weights = True)

    torch.save({'pou': static_neural_field.psi,
                'C': static_neural_field.C,
    }, args.out_folder + 'static_net.pt')

    #Forms Imaging systems and targets for each batch
    spatial_to_dynamic(dynamic_neural_field, static_neural_field)
    
    
    print(sep, "Dynamic Component", sep)
    dynamic_targets = get_dynamic_targets(measurements, radii, args.ntB, args.nrB, args.nT, args.vpt)
    #List of regularization parameters
    deltas = [2, 1, 1/2, 1/4, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 0] 
    alpha = 1
    for delta in deltas:
        print("Regularization Paramater = ", delta)
        inner_thresholdC = 1e-2
        inner_threshold_net = 1e-1
        
        train_network(dynamic_neural_field, dynamic_systems, dynamic_targets,\
                        delta, std, args.nI, inner_thresholdC, inner_threshold_pou, dev)
        alpha = 0
        for dynamic_sys, dynamic_target in zip(dynamic_systems, dynamic_targets):
            torch.cuda.empty_cache()
            alpha += data_fidelity(dynamic_sys.B.to(dev), dynamic_neural_field, dynamic_sys.X.to(dev), dynamic_target.to(dev), std).detach()/dynamic_nB
        print("alpha = ", alpha)
        
        #Saves Network 
        torch.save({'pou': dynamic_neural_field.psi,
                    'C': dynamic_neural_field.C,
        }, args.out_folder + 'dynamic_net.pt')
        if alpha <= 0.51:
            break