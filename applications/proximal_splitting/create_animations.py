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
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.metrics import structural_similarity as ssim

import sys
import argparse
sys.path.append( "../../")
from nfCRT import *

def save_animation(image, time_int, filename):
    """Saves image to file with time labels

    Args:
        image (numpy array): space time image
        time_int (list): list of times
        filename (string): file name to be saved in
    """
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', '5%', '5%')
    im = ax.imshow(image[:,:,0], cmap = 'gray', origin = 'lower')
    cb = fig.colorbar(im,cax = cax)

    plt.tick_params(
        axis='both',          # changes apply to the x and y-axis
        which='both',      # both major and minor ticks are affected
        left = False,
        right = False,
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelleft = False,
        labelbottom=False) # labels along the bottom edge are off

    tx = ax.set_title('Frame 0', size = 30)

    vmax = np.max(image)
    vmin = np.min(image)

    def animate(i):
        arr = image[:,:,i]
        im.set_data(arr)
        im.set_clim(vmin,vmax)
        t = int(100*(time_int[i] + max(time_int)) )/100
        tx.set_text('Time ' + str(t))
    ani = animation.FuncAnimation(fig, animate, save_count = image.shape[2])
    ani.save(filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create animations')
    parser.add_argument('--out_folder',
                        default= 'Reconstruction_Results/',
                        type = str)
    args = parser.parse_args()
    if torch.cuda.is_available():
        print("Cuda available, Implementing with GPU")
        dev = torch.device('cuda:0')
    else:
        print("Cuda not available, Implementing with CPU")
        dev = torch.device('cpu')

    nV = 72
    nrevs = 18
    image = io.loadmat('induced_pressure')['p0_time'][:,:,nV*0:(nrevs + 0)*nV]
    anat = io.loadmat('anatomy.mat')['phan_time'][:,:,nV*0:(nrevs + 0)*nV] == 9
    s = image.shape[0]
    nA = image.shape[2]
    
    time_int = nrevs*(torch.linspace(-1/2, 1/2,nA+1)[:-1] + 1/(2*nA))
    x_int = torch.linspace(-1/2,1/2,s+1)[:-1] + 1/(2*s)
    t, x0, x1 =torch.meshgrid(time_int, x_int, x_int)
    x0 = x0.flatten()
    x1 = x1.flatten()
    t = t.flatten()
    x = torch.stack((x0, x1, t))
    x = torch.transpose(x,0,1).float().to(dev)

    save_animation(image, time_int, args.out_folder + 'Phantom.gif')
    image_vec = image.flatten('F')
    net = torch.load(args.out_folder + 'dynamic_net.pt')

    dpou = net['pou']

    dC = net['C']
    
    out = torch.zeros(x.size()[0]).to(dev)
    
    N = s**2
    n_render = 180
    for i in range(n_render):
        inds = np.arange(i,N*nA, n_render)
        space_inds = inds%(s**2)
        psi = dpou(x[inds,-1:])
        out[inds] = torch.einsum('ij,ij->i', psi, dC[space_inds,:]).detach()

    out_image = out.detach().cpu().numpy().reshape((s,s,nA), order = 'F')
    error = out_image - image
    err_norm = np.linalg.norm(error)


    plt.clf()
    image_curve = np.sum(anat*image, axis = (0,1))/np.sum(anat, axis = (0,1))
    learned_curve =np.sum(anat*out_image, axis = (0,1))/np.sum(anat, axis = (0,1))

    plt.plot(image_curve, label = 'true')
    plt.plot(learned_curve, label = 'learned')
    plt.legend()
    plt.savefig(args.out_folder + 'lesion_intensity_curve.png')

    curve_error = np.linalg.norm(image_curve - learned_curve)
    print("Curve Error = ", curve_error)

    
    print("RRMSE = ", err_norm/np.linalg.norm(image_vec))
    print("Error = ", err_norm)
    print("Norm of Image = ", np.linalg.norm(image_vec))
    
    
    SSIM = ssim(image, out_image, data_range = out_image.max()-out_image.min())
    print("SSIM = ", SSIM)

    

    save_animation(out_image, time_int, args.out_folder + "Reconstructed.gif" )
    io.savemat(args.out_folder + "Reconstructed.mat", {"out_image": out_image})
    save_animation(error  , time_int, args.out_folder + "Error.gif" )
    
    Cn = dC.cpu().detach().numpy()
    plt.clf()
    plt.imshow(np.abs(Cn), aspect = 'auto')
    plt.colorbar()
    plt.savefig(args.out_folder + "dynamic_coeff_magnitudes.png")

    
    x = time_int.reshape(-1,1)
    dynamic_parition = np.zeros((nA,dpou.N_part))
    n_render = 1
    for j in range(n_render):
        dynamic_parition[j::n_render,:] = dpou(x[j::n_render,-1:].to(dev)).cpu().detach().numpy()
    time_int = nrevs*(np.linspace(-1/2, 1/2,nA+1)[:-1] + 1/(2*nA))
    plt.clf()
    for j in range(dpou.N_part):
        plt.plot(time_int, dynamic_parition[:,j], linewidth = 5)

    plt.savefig(args.out_folder + "partition.png")
    io.savemat(args.out_folder + "partition.mat", {"partitions": dynamic_parition})
