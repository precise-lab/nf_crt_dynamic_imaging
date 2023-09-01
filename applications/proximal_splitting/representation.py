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
    dev = x.get_device()
    out = torch.zeros(x.size(0),N).to(dev)

    rl = torch.nn.ReLU()
    
    for j in range(N):
        xj = 18*j/(N-1)-9
        out[:,j] = rl( 1- (N-1)*torch.abs(x-xj)/18)
    return out
def initialize_paritions(dpou, dev):
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
    parser.add_argument('--nB',
                        default = 288,
                        type = int)
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

    s = image.shape[0]
    nA = image.shape[2]



    N = s**2

    #POU will contain wsq^3 functions
    geodim = 1
    Width_dynamic = 140
    number_of_partitions = 10
    Depth = 4
    dynamic_pou = POU_Siren(Width_dynamic, number_of_partitions, Depth = Depth, geodim = geodim).to(dev)
    C = torch.zeros((N,number_of_partitions), device = dev)
    for j  in range(number_of_partitions):
        C[:,j] = torch.from_numpy( np.mean(image[:,:,130*j:130*(j+1)],2).flatten('F')).to(dev)
    nf = PwNeuralField(dynamic_pou, C)

    image = image.flatten('F')
    image_torch =  torch.from_numpy(image).float().to(dev)
    

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
    nb = args.nB

    Lrmax = 1e-4
    Lrmin = 1e-6
    image_norm = np.sum(image**2)**(1/2)

    ntimes = nA

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
                optimizer1.zero_grad()
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
        a = torch.zeros(N,number_of_partitions,number_of_partitions).to(dev)
        b = torch.zeros(N,number_of_partitions).to(dev)
        n_render = 16

        with torch.no_grad():
            for f in range(0,nA,n_render):
                INDS = inds[f::nA]
                for k in range(1,n_render):
                    INDS = np.append(INDS, inds[f+k::nA])
                y = x[INDS,:].to(dev)
                PSI = dynamic_pou(y[:,-1:])
                af = PSI[:,:,None]*PSI[:,None,:]

                for k in range(n_render):
                    a += af[k*N:(k+1)*N,:,:]
            
                target = image_torch[INDS].to(dev)
                psit = PSI*target[:,None]
                

                for k in range(n_render):
                    b += psit[k*N:(k+1)*N,:]
        a = a.cpu().detach().numpy().flatten()
        b = b.cpu().detach().numpy().flatten()
  
        A = sp.csr_matrix( (a, (Arow,Acol)), shape = (number_of_partitions*N, number_of_partitions*N))
        Cnp, exit_code   = sp.linalg.cg(A,b, x0 = C.flatten().cpu().detach().numpy(), atol = 0, tol = 1e-6, maxiter = 100)#, M = M)
        C[:,:] = torch.from_numpy(Cnp).to(dev).reshape(N,number_of_partitions)
        
        with torch.no_grad():
            out_image = np.zeros(s**2*nA)
            for b in range(nb):
                inds = np.arange(b,N*nA, nb)
                space_inds = inds%(s**2)
                y = x[inds,:].to(dev)
                out = nf(y[:,-1:], space_inds)
        err = np.linalg.norm(out_image - image)/np.linalg.norm(image)
        print("     RRMSE = ", err)
        torch.save({'pou': dynamic_pou,
                    'C': C,
        }, out_folder + 'dynamic_net.pt')




   