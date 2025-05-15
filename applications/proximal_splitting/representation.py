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
sys.path.append( "../../")

import torch
import random
import numpy as np
import scipy.io as io
import scipy.sparse as sp
import argparse

from nfCRT import *


def cg_basis(N,x):
    """Forms continuous Galerkin basis for initializing partitions
    
    Args:
        N (int): number of interpolation funcitons
        x (pytorch tensor): points to evaluate CG basis

    Returns:
        tensor: value of CG basis evaluated at selected points
    """
    dev = x.get_device()
    out = torch.zeros(x.size(0),N).to(dev)

    rl = torch.nn.ReLU()
    
    for j in range(N):
        xj = 18*j/(N-1)-9
        out[:,j] = rl( 1- (N-1)*torch.abs(x-xj)/18)
    return out
    
def initialize_paritions(dpou, dev):
    """Solves optimization problem to initialize partition of unity(POU) to CG basis

    Args:
        dpou (partition of unity): partition of unity  to be initialized
        dev (pytorch device): device name for evaluation
    """
    print("Initializing Partitions")

    opt = torch.optim.Adam(dpou.parameters(), lr = 1e-4)

    N = dpou.N_part

    Np = 10**4
    ni = 10**3

    for i in range(ni):
        i/ni
        opt.zero_grad()

        x = torch.randn((Np,3), device = dev) - 1/2
        x[:,-1] *= 18

        target = cg_basis(N,x[:,-1])
        loss = torch.mean(( dpou(x[:,-1:])-target)**2)/2
        loss.backward()
        opt.step()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Supervised training to solve representaion problem')
    parser.add_argument('--out_folder',
                        default= 'Representation_Results/',
                        type = str)
    args = parser.parse_args()
    out_folder = args.out_folder

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
    
    nrevs = 18
    image = io.loadmat('induced_pressure.mat')['p0_time'][:,:,72*0:72*(nrevs+ 0)]

    image_time = torch.from_numpy(np.sum(image, axis = (0,1))).to(dev)

    s = image.shape[0]
    nA = image.shape[2]



    N = s**2
    geodim = 1
    Width_dynamic = 140
    number_of_partitions = 10
    Depth = 4
    dynamic_pou = POU_Siren(Width_dynamic, number_of_partitions, Depth = Depth, geodim = geodim).to(dev)
    C = torch.zeros((N,number_of_partitions), device = dev)

    for j  in range(number_of_partitions):
        C[:,j] = torch.from_numpy( np.mean(image[:,:,130*j:130*(j+1)],2).flatten('F')).to(dev)
    image_torch =  torch.from_numpy(image.flatten('F')).float().to(dev)
    image_rec = torch.from_numpy(image.reshape((s**2,nA), order = 'F')).float().to(dev)
    

    time_int = nrevs*(torch.linspace(-1/2, 1/2,nA+1)[:-1] + 1/(2*nA))
    x_int = torch.linspace(-1/2,1/2,s+1)[:-1] + 1/(2*s)

    t, x0, x1 =torch.meshgrid(time_int, x_int, x_int)
    x0 = x0.flatten()
    x1 = x1.flatten()
    t = t.flatten()
    x = torch.stack((x0, x1, t))
    x = torch.transpose(x,0,1).float()

    initialize_paritions(dynamic_pou, dev)
    torch.save({'pou': dynamic_pou,
                    'C': C,
        }, out_folder + 'dynamic_net.pt')

    ni = 1
    nI = 100
    nb = 16*nrevs

    image_norm = np.sum(image**2)**(1/2)

    row_inds, col_inds = torch.meshgrid(torch.arange(number_of_partitions), torch.arange(number_of_partitions))

    Arow = [] 
    Acol = []
    for j in range(N):
        #print(j/N)
        Arow += list((j*number_of_partitions + row_inds.flatten()).detach().numpy())
        Acol += list((j*number_of_partitions + col_inds.flatten()).detach().numpy())


    N = s**2
    for I in range(nI+1):
        torch.cuda.empty_cache()

        Lr = Lrmin + (nI - I)*(Lrmax - Lrmin)/nI 
        #Lr = Lrmax*(Lrmin/Lrmax)**(I/nI)
        optimizer = torch.optim.Adam(dynamic_pou.parameters(), lr = Lr)
        for i in range(ni):
            acum_loss = 0
            for b in range(nb):
                optimizer.zero_grad()
                inds = np.arange(b,N*nA, nb)

                space_inds = inds%(s**2)
                y = x[inds,:].to(dev)
                out = nf(y[:,-1:], space_inds)

                loss = torch.sum((out - image_torch[inds])**2)/2
                

                loss.backward()
                optimizer.step()


                acum_loss += loss.cpu().detach().numpy()
        print("Loss after Training Network = ", acum_loss)
        inds = ((np.arange(N)).reshape((-1,1)) + s**2*np.arange(nA).reshape(1,-1)).flatten()

        y = time_int.reshape(-1,1).to(dev)
        psi = dynamic_pou(y)
        a = torch.sum(psi[:,:,None]*psi[:,None,:],dim = 0)
        a = torch.stack([a]*N, dim = 0)#.cpu().detach().numpy().flatten()
        
        b = torch.zeros(N,number_of_partitions).to(dev)
        n_render = 16

        with torch.no_grad():
            for f in range(0,nA,n_render):
                INDS = inds[f::nA]
                for k in range(1,n_render):
                    INDS = np.append(INDS, inds[f+k::nA])
                target = image_torch[INDS].to(dev)
                psit = psi[INDS//N,:]*target[:,None]
                for k in range(n_render):
                    b += psit[k*N:(k+1)*N,:]

        
        a = a.cpu().detach().numpy().flatten()
        b = b.cpu().detach().numpy().flatten()
  
        A = sp.csr_matrix( (a, (Arow,Acol)), shape = (number_of_partitions*N, number_of_partitions*N))
        #ml = pyamg.smoothed_aggregation_solver(A)
        #M = ml.aspreconditioner()
        Cnp, exit_code   = sp.linalg.cg(A,b, x0 = C.flatten().cpu().detach().numpy(), atol = 0, tol = 1e-6, maxiter = 100)#, M = M)
        C[:,:] = torch.from_numpy(Cnp).to(dev).reshape(N,number_of_partitions)
        
        torch.cuda.empty_cache()

        Lr = 1e-4
        ni = 10**3
        with torch.no_grad():
            c = torch.mean(C[:,:,None]*C[:,None,:],dim = 0)
            L, V = np.linalg.eig(c.cpu().detach().numpy())

            A_pos = torch.from_numpy(np.diag(L**(1/2))@V.T).to(dev)
            A_neg = torch.from_numpy(np.diag(L**(-1/2))@V.T).to(dev)
            b = torch.mean( image_rec[:,None,:]*C[:,:,None], dim = 0)
            
            b = A_neg@b
        opt = torch.optim.Adam(dynamic_pou.parameters(), lr = Lr)
        for i in range(ni):
            opt.zero_grad()
            psi = dynamic_pou(time_int.to(dev).reshape(-1,1))
            out = A_pos@psi.T
            loss = torch.sum((out - b)**2)/2
            loss.backward()
            opt.step()

        
        with torch.no_grad():
            out_image = np.zeros(s**2*nA)
            for b in range(nb):
                inds = np.arange(b,N*nA, nb)
                space_inds = inds%(s**2)
                y = x[inds,:].to(dev)
                psi = dynamic_pou(y[:,-1:])
                out_image[inds] = ( torch.einsum('ij,ij->i', psi, C[space_inds,:])).cpu().detach().numpy()
        err = np.linalg.norm(out_image - image.flatten('F'))/np.linalg.norm(image)
        print("     RRMSE = ", err)
        torch.save({'pou': dynamic_pou,
                    'C': C,
        }, out_folder + 'dynamic_net.pt')




   