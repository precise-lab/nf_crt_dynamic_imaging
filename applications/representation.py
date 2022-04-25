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

import sys
import os
import argparse
sys.path.append( "../")

import torch
import random
import numpy as np
import scipy.io as io

from nfCRT import *

def rep_loss(nf, X, measurements, \
                    gamma, rho):
    """Computes Representation Loss

    Args:
        nf (NeuralField): neural field
        X (pytorch tensor): location for comparison points
        measurements (pytorch tensor): target
        gamma (double): l_{1/2} training regularization weight on partition
        rho (_type_): Frobenius training regularization weight on coefficients

    Returns:
        _type_: _description_
    """
    
    OUT = nf(X)
    sq = (OUT - measurements)**2
    loss = (1/2)*torch.sum(sq)
    if gamma > 0:
            loss += gamma*torch.mean( torch.sum((nf.psi(X) + 0.01)**(1/2),1)**2)
    if rho > 0:
        loss += rho*torch.mean(nf.C**2)
    return loss

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

    parser = argparse.ArgumentParser(description='Invert for Static average.')
    parser.add_argument('--out_folder',
                        default= 'Representation_Results/',
                        type = str)
    parser.add_argument('--nB',
                        default = 4,
                        type = int)
    args = parser.parse_args()
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
    
    image = io.loadmat('initial_pressure')['p0']
    s = image.shape[0]
    nA = image.shape[2]

    ave_image = np.sum(image,2)/nA

    print('ave_image size = ', ave_image.shape)

    #Set of Dynamic Coordinates
    time_int = torch.linspace(-1/2, 1/2,nA)
    x_int = torch.linspace(-1/2,1/2,s)
    x1, x0, t =torch.meshgrid(x_int, x_int, time_int)
    x0 = x0.flatten()
    x1 = x1.flatten()
    t = t.flatten()
    x = torch.stack((x0, x1, t))
    x = torch.transpose(x,0,1).float().to(dev)

    #Set of Static Coordinates
    y1, y0 =torch.meshgrid(x_int, x_int)
    y0 = y0.flatten()
    y1 = y1.flatten()
    y = torch.stack((y0, y1))
    y = torch.transpose(y,0,1).float().to(dev)
    
    #POU will contain wsq^3 functions
    geodim = 3
    Width_static = 100
    Width_dynamic = 140
    number_of_partions_static = 20
    number_of_partions_dynamic = 40
    Depth = 4
    dynamic_pou = POU_Siren(Width_dynamic, number_of_partions_dynamic, Depth = Depth, geodim = geodim).to(dev)
    static_pou = POU_Siren(Width_static, number_of_partions_static, Depth = Depth, geodim = geodim - 1).to(dev)
    
    mx = 3
    mt = 3
    dynamic_P = Polynomial(geodim, mx, mt = mt, st_separability = True )
    static_P = Polynomial(geodim- 1, mx, st_separability = False )
    
    dynamic_neural_field = NeuralField(dynamic_pou, dynamic_P, dev)
    static_neural_field = NeuralField(static_pou, static_P, dev)
    

    sep = "\n"+"#"*80+"\n"
    
    #Initial training regularization weight
    rho = 1e2
    gamma = 1e2

    nI = 20
    ni = 20
    
    weight_shrinkage = (0.1)**(4/nI)

    Lrmax = 1e-3
    Lrmin = 1e-6
    
    static_target = torch.from_numpy(ave_image.flatten()).float().to(dev)
    
    def static_closure(optimizer):
                optimizer.zero_grad()
                loss = rep_loss(static_neural_field, y, static_target, \
                            gamma, rho)
                loss.backward()
                return loss
    LrC = 1e-3
    for i in range(nI + 1):
        if i < nI/2:
            rho *= weight_shrinkage
            gamma *= weight_shrinkage
        else:
            LrC = 1e-4
            rho = 0
            gamma = 0
        print(sep, "Average Recon: Outer Iteration", i,  sep)
        Lr = Lrmin + (nI - i)*(Lrmax - Lrmin)/nI
        otpimizer0 = optimizer0 = torch.optim.Adam([static_neural_field.C], lr = LrC)
        def closure0():
                return static_closure(optimizer0)
        optimizer1 = torch.optim.Adam(static_neural_field.psi.parameters(), lr = Lr)
        def closure1():
                return static_closure(optimizer1)
        torch.cuda.empty_cache()
        for k in range(ni):
            optimizer0.step(closure0)
        loss = closure0()
        print("Loss after Updating Coefficients = ", loss.cpu().detach().numpy())
        torch.cuda.empty_cache()
        for k in range(ni):
            optimizer1.step(closure1)
        loss = closure1()
        print("Loss after Training Network = ", loss.cpu().detach().numpy())
        
    torch.save({'pou': static_neural_field.psi,
                'C': static_neural_field.C,
    }, args.out_folder +'static_net.pt')
    
    spatial_to_dynamic(dynamic_neural_field, static_neural_field)
    target =  torch.from_numpy(image.flatten()).float().to(dev)

    rho = 0
    gamma = 0
    nB = 4

    def dynamic_closure(optimizer, j):
                optimizer.zero_grad()
                loss = rep_loss(dynamic_neural_field, x[j::nB, :], target[j::nB], \
                            gamma, rho)
                loss.backward()
                return loss
    LrC = 1e-3
    for i in range(nI + 1):

        print(sep, "Dynamic Recon: Outer Iteration", i,  sep)
        Lr = Lrmin + (nI - i)*(Lrmax - Lrmin)/nI
        otpimizer0 = optimizer0 = torch.optim.Adam([dynamic_neural_field.C], lr = LrC)
        optimizer1 = torch.optim.Adam(dynamic_neural_field.psi.parameters(), lr = Lr)
        torch.cuda.empty_cache()
        acum_loss = 0
        
        for k in range(ni):
            for j in range(nB):
                def closure0():
                    return dynamic_closure(optimizer0, j)
                optimizer0.step(closure0)
        loss = closure0()
        acum_loss += loss
        print("Loss after Updating Coefficients = ", acum_loss)
        acum_loss = 0
        torch.cuda.empty_cache()

        for k in range(ni):
            for j in range(nB):
                def closure1():
                    return dynamic_closure(optimizer1, j)
                optimizer1.step(closure1)
        loss = closure1()
        acum_loss += loss
        print("Loss after Training Network = ", acum_loss)
        
        torch.save({'pou': dynamic_neural_field.psi,
                        'C': dynamic_neural_field.C,
            }, args.out_folder + 'dynamic_net.pt')