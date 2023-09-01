import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.io as io
import scipy.sparse as sp
import copy
import pyamg

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

    Np = 10**3
    ni = 10**4

    for i in range(ni):
        opt.zero_grad()

        #x = torch.zeros((Np,3), device = dev)
        #x[:,:-1] = 1/4*torch.randn((Np,2), device = dev)
        #x[:,-1] = 18*(torch.rand(Np, device = dev) - 1/2)

        x = 18*(torch.rand((Np,1), device = dev) - 1/2)
        

        target = cg_basis(x[:,0])
        loss = torch.mean(( dpou(x)-target)**2)/2
        loss.backward()
        opt.step()



def form_systems(vpt, cover, rnl):
    #image = np.load('../initial_pressure.npy')
    global nrevs
    global nV
    nV = 72
    global image
    image = io.loadmat('p0_z_280_time.mat')['p0_time'][:,:,nV*0:nV*(nrevs+0)]
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
    
        #pc = (nR*np.array(angles_per_view_inds[i%nV]).reshape(vpt,1) + np.arange(nR).reshape(1,nR)).flatten()
        pc = (nR*np.array(angles_per_view_inds[i%nV]).reshape(vpt*cover,1)+ np.arange(nR).reshape(1,nR)).flatten()
        Pa = a[pc,:]

        crt_ops.append(Pa)
        target[nR*vpt*cover*i:nR*vpt*cover*(i+1)] = Pa@image[:,:,i].flatten('F')
    
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

def form_derivative_matrices():
    global s
    global number_of_partitions
    global nA
    N = s**2

    row_inds, col_inds = torch.meshgrid(torch.arange(number_of_partitions), torch.arange(number_of_partitions))

    global Arow
    global Acol


    Arow = [] 
    Acol = []
    for j in range(N):
        Arow += list((j*number_of_partitions + row_inds.flatten()).detach().numpy())
        Acol += list((j*number_of_partitions + col_inds.flatten()).detach().numpy())

    global inds
    global indsx
    global indsy

    inds = ((np.arange(N)).reshape((-1,1)) + s**2*np.arange(nA).reshape(1,-1)).flatten()
    indsx = ((s*np.arange(s-1).reshape(-1,1) + np.arange(s).reshape(1,-1)).reshape(-1,1) + s**2*np.arange(nA).reshape(1,-1)).flatten()
    indsy = ((s*np.arange(s).reshape(-1,1) + np.arange(s-1).reshape(1,-1)).reshape(-1,1) + s**2*np.arange(nA).reshape(1,-1)).flatten()

    global Dx
    global Dy

    rx = []
    cx = []
    vx = []

    for j in range(s-1):
        for i in range(s):
            for l in range(number_of_partitions):
                rx.append(j*s*number_of_partitions+i*number_of_partitions + l)
                rx.append(j*s*number_of_partitions+i*number_of_partitions + l)

                cx.append((j+1)*s*number_of_partitions + i*number_of_partitions + l)
                cx.append(j*s*number_of_partitions + i*number_of_partitions + l)

                vx.append(1)
                vx.append(-1)
    Dx = sp.csr_matrix((vx, (rx, cx)), shape = ((s-1)*s*number_of_partitions, s**2*number_of_partitions ))
    Dx = Dx.astype(np.int8)

    ry = []
    cy = []
    vy = []

    for j in range(s):
        for i in range(s-1):
            for l in range(number_of_partitions):
                ry.append(j*(s-1)*number_of_partitions+i*number_of_partitions + l)
                ry.append(j*(s-1)*number_of_partitions+i*number_of_partitions + l)

                cy.append(j*s*number_of_partitions + (i+1)*number_of_partitions + l)
                cy.append(j*s*number_of_partitions + i*number_of_partitions + l)

                vy.append(1)
                vy.append(-1)
    Dy = sp.csr_matrix((vy, (ry, cy)), shape = ((s-1)*s*number_of_partitions, s**2*number_of_partitions))
    Dy = Dy.astype(np.int8)

def coefficient_update(dpou, C, old_pou, old_C, x, LrD, delta):
    dev = C.get_device()

    global s
    global nA
    global number_of_partitions
    global Arow
    global Acol

    global inds
    global indsx
    global indsy

    N = s**2

    a = torch.zeros(N,number_of_partitions,number_of_partitions).to(dev)
    a_xhalf = torch.zeros(s*(s-1),number_of_partitions,number_of_partitions).to(dev)
    a_yhalf = torch.zeros(s*(s-1),number_of_partitions,number_of_partitions).to(dev)

    b = torch.zeros(N,number_of_partitions).to(dev)

    n_render = 12

    with torch.no_grad():
        torch.cuda.empty_cache()
        for f in np.arange(0,nA,n_render):
        
            INDS = inds[f::nA]
            INDSX = indsx[f::nA]
            INDSY = indsy[f::nA]

            for k in range(1,n_render):
                INDS = np.append(INDS, inds[f+k::nA])
                INDSX = np.append(INDSX, indsx[f+k::nA])
                INDSY = np.append(INDSY, indsy[f+k::nA])

            y = x[INDS,:].to(dev)
            PSI = dpou(y[:,-1:])

            af = torch.einsum('ik,il->ikl', PSI, PSI)
            for k in range(n_render):  
                a += af[k*N:(k+1)*N,:,:]

            
            target = torch.einsum('ij,ij ->i', old_pou(x[INDS,-1:].to(dev)), old_C[INDS%N,:]) - LrD*Grad(INDS)
            outb = (PSI*target[:,None])

            for k in range(n_render): 
                b += outb[k*N:(k+1)*N,:]

            y =  x[INDSX,:].to(dev)
            PSI = dpou(y[:,-1:])
            outa_xhalf = (LrD*delta/(s*(s-1)*nA))*torch.einsum('ik, il-> ikl',PSI,PSI)
            for k in range(n_render):
                a_xhalf += outa_xhalf[k*s*(s-1):(k+1)*s*(s-1),:,:]

            y = x[INDSY,:].to(dev)
            PSI = dpou(y[:,-1:])
            outa_yhalf = (LrD*delta/(s*(s-1)*nA))*torch.einsum('ik, il-> ikl',PSI,PSI)
            for k in range(n_render):
                
                a_yhalf += outa_yhalf[k*s*(s-1):(k+1)*s*(s-1),:,:]
    a = a.cpu().detach().numpy().flatten()
    b = b.cpu().detach().numpy().flatten()


    a_xhalf = a_xhalf.cpu().detach().numpy().flatten()
    a_yhalf = a_yhalf.cpu().detach().numpy().flatten()

                

    A = sp.csr_matrix( (a, (Arow,Acol)), shape = (number_of_partitions*N, number_of_partitions*N), dtype = np.single)
    A_xhalf = sp.csr_matrix( (a_xhalf, (Arow[:-s*number_of_partitions**2],Acol[:-s*number_of_partitions**2])), shape = (number_of_partitions*s*(s-1), number_of_partitions*s*(s-1)), dtype =  np.single)
    A_yhalf = sp.csr_matrix( (a_yhalf, (Arow[:-s*number_of_partitions**2],Acol[:-s*number_of_partitions**2])), shape = (number_of_partitions*s*(s-1), number_of_partitions*s*(s-1)), dtype = np.single)

    Afull = A + Dx.T@A_xhalf@Dx + Dy.T@A_yhalf@Dy
    #ml = pyamg.smoothed_aggregation_solver(Afull)
    #M = ml.aspreconditioner().astype(Afull.dtype)
    
    Cnp, exit_code   = sp.linalg.cg(Afull,b, x0 = C[:,:].flatten().cpu().detach().numpy(), tol = 1e-6, atol = 0, maxiter = 20)
    
    C[:,:] = torch.from_numpy(Cnp).to(dev).reshape(N,number_of_partitions)
    
def nf_proximal_update_tv_reg(dpou, C, old_pou, old_C, Grad, x, LrD, delta):
    dev = C.get_device()

    global s
    global nA
    global number_of_partitions
    global nrevs

    nI = 1
    ni = 100
    nb = 1

    Np = 10**4
    N = s**2

    global global_it
    Lr = 1e-5#1e-1*LrD#*(0.99)**(global_it)


    for I in np.arange(nI):
        coefficient_update(dpou, C, old_pou, old_C, x, LrD, delta)
        
        optimizer1 = torch.optim.Adam(dpou.parameters(), lr = Lr)
        global inds
        global indsx
        global indsy

        for i in range(ni):

            for b in range(nb):
        
                optimizer1.zero_grad()
                inds0 = np.random.randint(N*nA, size = (Np,))
                indsx0 = s*np.random.randint(s-1, size = (Np,)) + np.random.randint(s, size = (Np,)) + s**2*np.random.randint(nA, size = (Np,))
                indsy0 = s*np.random.randint(s, size = (Np,)) + np.random.randint(s-1, size = (Np,)) + s**2*np.random.randint(nA, size = (Np,))

                space_inds = inds0%(s**2)
                space_indsx = indsx0%(s**2)
                space_indsy = indsy0%(s**2)

                y = x[inds0,:].to(dev)
                #y.requires_grad = True

                psi = dpou(y[:,-1:])
                out = torch.einsum('ij,ij->i', psi, C[space_inds,:])

                target =  torch.einsum('ij,ij ->i', old_pou(y[:,-1:]), old_C[space_inds,:])  - LrD*Grad(inds0)
                
        

                loss = torch.mean((out - target)**2)/(2*LrD)

                if delta > 0:
                    psi_xhalf = torch.einsum('ij,ij->i', dpou( (x[indsx0,-1:] ).to(dev)), C[space_indsx+s,:]  - C[space_indsx,:]  ) 
                    psi_yhalf = torch.einsum('ij,ij->i', dpou( (x[indsy0,-1:] ).to(dev)), C[space_indsy + 1,:]  - C[space_indsy,:]  )
                    reg = (delta/2)*(torch.mean(psi_xhalf**2 + psi_yhalf**2))/(N*nA)
                    loss += reg
                
                loss.backward()
                optimizer.step() 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image reconstruction using proximal splitting.')
    parser.add_argument('--out_folder',
                        default=  'Reconstruction_Results/',
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
    
    #Creates measurements, and imaging systems
    a, imaging_op, angle_inds, ave_meas, measurements, std = form_systems(args.vpt, args.cover, args.rnl)

    np.save(args.out_folder + 'meas', measurements)
    np.save(args.out_folder + 'ave_meas', ave_meas)
    


    #Geometric Dimension
    geodim = 1
    
    Width = 140

    number_of_partitions = 10
    
    Depth = 4

    
        
    print("Creating New Model")
    dpou = POU_Siren(Width, number_of_partitions, geodim = geodim, Depth = Depth).to(dev)
    C = torch.zeros((s**2,number_of_partitions), device = dev)


    ave_image = np.zeros((s**2,number_of_partitions))
    image = io.loadmat('induced_pressure.mat')['p0_time'][:,:,nV*0:nV*(nrevs+0)]

    AVE_meas = sp.linalg.lsqr(a, np.mean(ave_meas,1), atol = 1e2*std/args.cover)[0]

    for j in range(number_of_partitions -1):
        ave_image[:,j+1] =  sp.linalg.lsqr(a, ave_meas[:,j], atol = 2e2*std/args.cover)[0]
        
        plt.imshow(ave_image[:,j+1].reshape((s,s),order = 'F'))
        plt.colorbar()
        plt.savefig(args.out_folder + 'average_recon{}.png'.format(j))
        plt.clf()

        plt.imshow(ave_image[:,j+1].reshape((s,s),order = 'F')-np.mean(image[:,:,j*144:(j+1)*144],2))
        plt.colorbar()
        plt.savefig(args.out_folder + 'diff{}.png'.format(j))
        plt.clf()

        plt.imshow(np.mean(image[:,:,j*144:(j+1)*144],2))
        plt.colorbar()
        plt.savefig(args.out_folder + 'true{}.png'.format(j))
        plt.clf()

    ave_image[:,0] = ave_image[:,1]

    


    initialize_paritions(dpou, dev)
    for j in  range(number_of_partitions):
        C[:,j] = torch.from_numpy(ave_image[:,j]).detach().to(dev)
        
    torch.save({'pou': dpou,
        'C': C,
    }, args.out_folder + 'dynamic_net.pt')

    startit = 0
    
    form_derivative_matrices()

    
    x_int = torch.linspace(-1/2,1/2, s+1)[:-1]+1/(2*s)

    nB = args.nB
    global_it_max = 125*nB
    
    
    time_int = nrevs*(torch.linspace(-1/2, 1/2,nA+1)[:-1] + 1/(2*nA))
    x0, x1 =torch.meshgrid(x_int, x_int)
    y = torch.stack((x0.flatten(), x1.flatten(), torch.zeros(s**2)))
    y = torch.transpose(y,0,1).float()


    t, x0, x1 =torch.meshgrid(time_int, x_int, x_int)
    x = torch.stack((x0.flatten(), x1.flatten(), t.flatten()))
    x = torch.transpose(x,0,1).float()
    
    delta = 1e5/nB
    Np = 10**5
    
    

    n_report = 1

    dtf_list = np.zeros(int(global_it_max/n_report))
    err_list = np.zeros(int(global_it_max/n_report))
    cost_list = np.zeros(int(global_it_max/n_report))


    for global_it in range(startit, global_it_max):
        torch.cuda.empty_cache()

        old_C = C.clone().detach()
        old_part = copy.deepcopy(dpou)

            
        grad_dict = {}
        Lr_num = 0
        Lr_den = 0
        
        b = global_it%nB
        frame_inds = np.arange(b,nA,nB)
        grad_mag = 0
        for i in frame_inds:
            y[:,-1] = time_int[i]
            psi = dpou(y[:,-1:].to(dev))
            discrete_parameter = (torch.einsum('ij,ij->i',psi,C)).cpu().detach().numpy()
            Pa = imaging_op[i%nV]
            grad_dict[i] = (1/std**2)*np.transpose(Pa)@(Pa@discrete_parameter - measurements[i*args.vpt*args.cover*nR: (i+1)*args.vpt*args.cover*nR])
            bpg = Pa@grad_dict[i]
            Lr_den += np.inner(bpg, bpg)
            Lr_num += np.inner(grad_dict[i], grad_dict[i])
            grad_mag += Lr_den
        grad_mag **= 1/2
        print("Grad Mag ", grad_mag)
            
        
        Lr_disc = std**2*Lr_num/Lr_den


        def dcd(inp):
            int_locs = (inp%nB == b).nonzero()[0]
            out_locs = (inp[int_locs])//nB
            return int_locs, out_locs



        
        Grad = FrameBasedFunction(grad_dict, dev, decode_func = dcd)




        print("Learning rate: ", Lr_disc)
        

        M = np.prod(measurements.shape)
        nf_proximal_update_tv_reg(dpou, C, old_part, old_C, Grad, x, Lr_disc, delta*M)
        
        torch.save({'pou': dpou,
                    'C': C,
                    'global_it': global_it
        }, args.out_folder + 'dynamic_net.pt')

        if global_it%n_report ==0:

            df = 0
            err = 0
            
            for i in range(nA):
                y[:,-1] = time_int[i]
                discrete_parameter = (torch.einsum('ij,ij->i', dpou(y[:,-1:].to(dev)), C)).cpu().detach().numpy()
                
                
                err += np.sum( (discrete_parameter-image[:,:,i].flatten('F'))**2)
                df += np.sum( (imaging_op[i%nV]@discrete_parameter - measurements[i*nR*args.vpt*args.cover:(i+1)*nR*args.vpt*args.cover])**2)
            
            
            df /= M*std**2
            err /= np.sum(image**2)
            err **= 1/2



            if delta > 0:
                
                inds0 = np.random.randint(s**2*nA, size = (10**5,))
                indsx0 = s*np.random.randint(s-1, size = (Np,)) + np.random.randint(s, size = (Np,)) + s**2*np.random.randint(nA, size = (Np,))
                indsy0 = s*np.random.randint(s, size = (Np,)) + np.random.randint(s-1, size = (Np,)) + s**2*np.random.randint(nA, size = (Np,))
                space_inds = inds0%(s**2)
                
                z = x[inds0,:].to(dev)
                out =torch.einsum('ij,ij->i', dpou(z[:,-1:]), C[space_inds,:])


                psi_xhalf = torch.einsum('ij,ij->i', dpou( (x[indsx0,-1:].to(dev))), (C[indsx0%(s**2)+s,:]  - C[indsx0%(s**2),:]  ))
                psi_yhalf = torch.einsum('ij,ij->i', dpou( (x[indsy0,-1:].to(dev))), (C[indsy0%(s**2)+1,:]  - C[indsy0%(s**2),:]  ))
                reg = (delta/2)*(torch.mean(psi_xhalf**2)+torch.mean(psi_yhalf**2))
            else:
                reg = 0
            cost = 0.5*df + reg

            i = int(global_it/n_report)
            print("Iteration = ", i)
            print("     Data Fidelity = ", df)
            print("     RRMSE = ", err)
            print("     Cost = ", cost, "; Reg = ", reg)

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