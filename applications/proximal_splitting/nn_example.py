import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.io as io
import scipy.sparse as sp
import copy

import argparse
import sys
sys.path.append( "../../")
import os
from nfCRT import *

s = 0
nA = 0
nrevs = 18
nV = 0
nR = 0
number_of_partitions = 10

Dx = 0
Dy = 0
Arow = []
Acol = []

inds = 0
indsx = 0
indsy = 0

image = 0

beta = 1


def cg_basis(x):
    global number_of_partitions
    N = number_of_partitions
    out = np.zeros((x.shape[0], N))
    for j in range(N):
        xj = j/(N-1)-1/2
        out[:,j] = ( 1- (N-1)*np.abs(x-xj))
    out = out*(out > 0)

    return out


def form_systems(vpt, cover, rnl):
    global nrevs
    global nV
    nV = 72
    global image
    image = io.loadmat('induced_pressure.mat')['p0_time'][:,:,nV*0:nV*(nrevs+0)]
    global s
    s = image.shape[0]
    global nA
    nA = image.shape[-1]

    
    
    

    angles_per_view_inds = {}
    print("         Calculating views and times")
    for i in range(nV):
        angles_per_view_inds[i] = []
        for j in range(cover):
            for k in range(vpt):
                #angles_per_view_inds[i].append(  int( (i + (nA/vpt)*k)%nA))
                angles_per_view_inds[i].append(  int( (i*vpt + k + j*nV*vpt/cover)%(vpt*nV)))
    a = DiscreteCircularRadonTransform(s, angles = 2*np.arange(vpt*nV)*np.pi/(vpt*nV)).A

    global nR
    nR = int(a.shape[0]/(vpt*nV))
    
    
    
    crt_ops = []
    target = np.zeros(nR*nA*vpt*cover)
    for i in range(nA):
    
        if i < nV:
            pc = (nR*np.array(angles_per_view_inds[i%nV]).reshape(vpt*cover,1)+ np.arange(nR).reshape(1,nR)).flatten()
            Pa = a[pc,:]
            crt_ops.append(Pa)
        target[nR*vpt*cover*i:nR*vpt*cover*(i+1)] = crt_ops[i%nV]@image[:,:,i].flatten('F')
    
    std = rnl*np.max(target)
    print("Noise standard deviaiton", std)
    
    target +=  np.random.normal(0, std, len(target))
    global number_of_partitions
    ave_measurements = np.zeros((int(nR*nV*vpt), number_of_partitions-1))

    for r in range(number_of_partitions-1):
        for i in range(2*r*nV, 2*(r+1)*nV):
            for j, view in enumerate(angles_per_view_inds[i%nV]):
                ave_measurements[view*nR:(view+1)*nR,r] += number_of_partitions*target[( i*vpt*cover + j)*nR:(i*vpt*cover + j + 1)*nR]/(nrevs*cover)
                
                #3*(number_of_partitions-1)*target[( i*vpt*cover + j)*nR:(i*vpt*cover + j + 1)*nR]/(nrevs*cover)


    return a, crt_ops, angles_per_view_inds, ave_measurements, target,  std

  


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Matrix Free Implementation of CRT.')
    parser.add_argument('--out_folder',
                        default=  'InSilico_NN/',
                        type = str)
    parser.add_argument('--vpt',
                        default = 5,
                        type = int)
    parser.add_argument('--cover',
                        default = 2,
                        type = int)
    parser.add_argument('--nB',
                        default = 1,
                        type = int)
    parser.add_argument('--rnl',
                        default = 0.04,
                        type = float)
    parser.add_argument('--resume',
                        default = False,
                        type = bool)
    args = parser.parse_args()

    print('#'*20, args.vpt, " Views per time and ", args.rnl, " RNL", '#'*20)
    try:    
        os.mkdir(args.out_folder)
    except:
        pass

    
    #Creates measurements, and imaging systems
    a, imaging_op, angle_inds, ave_meas, measurements, std = form_systems(args.vpt, args.cover, args.rnl)

    np.save(args.out_folder + 'meas', measurements)
    np.save(args.out_folder + 'ave_meas', ave_meas)
    

    number_of_partitions = 10

    

    ave_image = np.zeros((s**2,number_of_partitions))
    for j in range(number_of_partitions -1):
        ave_image[:,j+1] =  sp.linalg.lsqr(a, ave_meas[:,j], atol = 2e2*std/args.cover)[0]     
    ave_image[:,0] = ave_image[:,1]



    startit = 0
    

    nB = args.nB
    global_it_max = 125*nB
    
    
    time_int = nrevs*(torch.linspace(-1/2, 1/2,nA+1)[:-1] + 1/(2*nA))
    
    delta = 1e-2/nB
    M = np.prod(measurements.shape)
    N = s**2
    
    n_report = 1

    dtf_list = np.zeros(int(global_it_max/n_report))
    err_list = np.zeros(int(global_it_max/n_report))
    cost_list = np.zeros(int(global_it_max/n_report))

    #partitions_over_its = np.zeros((nA,number_of_partitions, global_it_max))

    basis = cg_basis(time_int)
    recon = np.zeros((s**2, nA))
    for j in range(number_of_partitions):
        recon += ave_image[:,j:j+1]*basis[:,j].reshape((1,-1))



    for global_it in range(startit, global_it_max):

        old_recon = recon.copy()
        grad = np.zeros(recon.shape)

        Lr_num = 0
        Lr_den = 0
        
        b = global_it%args.nB
        frame_inds = np.arange(b,nA,args.nB)
        grad_mag = 0
        for i in frame_inds:   
            Pa = imaging_op[i%nV]
            grad[:,i] = (1/std**2)*np.transpose(Pa)@(Pa@recon[:,i]- measurements[i*args.vpt*args.cover*nR: (i+1)*args.vpt*args.cover*nR])
            bpg = Pa@grad[:,i]
            Lr_den += np.inner(bpg, bpg)
            Lr_num += np.inner(grad[:,i], grad[:,i])

        

        grad_mag = Lr_num**(1/2)
        print("Grad Mag ", grad_mag)
            
        
        Lr_disc = (std**2)*Lr_num/Lr_den


        recon -= Lr_disc*grad
        U, S, V = sp.linalg.svds(recon, k = 20)
        deltat = Lr_disc*delta*M
        St = (S- deltat)*(np.abs(S) > deltat)
        recon = U@(St[:,None]*V)
        rnk = np.sum(St > 0)

        io.savemat(args.out_folder + 'nn.mat', {'nn': recon})



        
        if global_it%n_report ==0:
            

            df = 0
            err = 0

            reg =  delta*np.sum(St)
            
            for i in range(nA):
        
                err += np.sum( (recon[:,i]-image[:,:,i].flatten('F'))**2)
                df += np.sum( (imaging_op[i%nV]@recon[:,i] - measurements[i*args.vpt*args.cover*nR: (i+1)*args.vpt*args.cover*nR])**2)
                
                
            df /= M*std**2
            err /= np.sum(image**2)
            err **= 1/2

            cost = 0.5*df + reg

            i = int(global_it/n_report)
            print("Iteration = ", i)
            print("     Data Fidelity = ", df)
            print("     RRMSE = ", err)
            print("     Cost = ", cost, "; Reg = ", reg, "; Rank = ", rnk)

            dtf_list[i] = df
            err_list[i] = err
            cost_list[i] = cost


            np.save(args.out_folder + 'data_fidelity',dtf_list[:i+1])
            plt.clf()
            plt.semilogy(dtf_list[:i+1])
            plt.savefig(args.out_folder + 'data_fidelity.png')

            np.save(args.out_folder + 'error',err_list[:global_it+1])
            plt.clf()
            plt.semilogy(err_list[:i+1])
            plt.savefig(args.out_folder + 'error.png')

            np.save(args.out_folder + 'cost',cost_list[:global_it+1])
            plt.clf()
            plt.semilogy(cost_list[:i+1])
            plt.savefig(args.out_folder + 'cost.png')
    