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
sys.path.append( "../")
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
        axis='both',          # changes apply to the x-axis
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
        t = int(100*(time_int[i] + 1/2) )/100
        tx.set_text('Time ' + str(t))
    ani = animation.FuncAnimation(fig, animate, save_count = image.shape[2])
    ani.save(filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create animations')
    parser.add_argument('--out_folder',
                        default= 'VPT4/',
                        type = str)
    args = parser.parse_args()
    if torch.cuda.is_available():
        print("Cuda available, Implementing with GPU")
        dev = torch.device('cuda:0')
    else:
        print("Cuda not available, Implementing with CPU")
        dev = torch.device('cpu')

    image = io.loadmat('initial_pressure')['p0']
    s = image.shape[0]
    nA = image.shape[2]
    
    time_int = torch.linspace(-1/2, 1/2,nA)
    x_int = torch.linspace(-1/2,1/2,s)
    x1, x0, t =torch.meshgrid(x_int, x_int, time_int)
    x0 = x0.flatten()
    x1 = x1.flatten()
    t = t.flatten()
    x = torch.stack((x0, x1, t))
    x = torch.transpose(x,0,1).float().to(dev)

    save_animation(image, time_int, args.out_folder + 'Phantom.gif')
    image_vec = image.flatten()
    net = torch.load(args.out_folder + 'dynamic_net.pt')
    ave_net = torch.load(args.out_folder + 'static_net.pt')

    dpou = net['pou']
    dC = net['C']
    ave_pou = ave_net['pou']
    ave_C = ave_net['C']

    mx = 3
    mt = 3
    geodim = 3
    P = Polynomial(geodim, mx, mt = mt, st_separability = True )
    ave_P = Polynomial(geodim -1, mx, st_separability = False)

    out = torch.zeros(x.size()[0]).to(dev)
    diff = torch.zeros(x.size()[0]).to(dev)
    ave = torch.zeros(x.size()[0]).to(dev)
    r = int(x.size()[0]/nA)
    for i in range(nA):
        y = x[r*i:r*(i+1),:]
        out[r*i:r*(i+1)] = torch.einsum( 'ij,jk,ki -> i', dpou(y), dC, P(y)).detach()
        ave[r*i:r*(i+1)] = torch.einsum( 'ij,jk,ki -> i', ave_pou(y[:,0:-1]), ave_C, ave_P(y[:,0:-1])).detach()
    diff = out - ave

    Cs = ave_C.cpu().detach().numpy()
    plt.clf()
    plt.imshow(np.abs(Cs))
    plt.colorbar()
    plt.savefig(args.out_folder + "static_coeff_magnitudes.png")
    
    print("RMSE = ", np.linalg.norm(image_vec - out.cpu().detach().numpy()  )/np.linalg.norm(image_vec))
    print("Error = ", np.linalg.norm(image_vec - out.cpu().detach().numpy()  ))
    print("Norm of Image = ", np.linalg.norm(image_vec))
    out_image = out.detach().cpu().numpy().reshape((s,s,nA))
    
    SSIM = ssim(image, out_image, dynamic_range = out_image.max()-out_image.min())
    print("SSIM = ", SSIM)
    diff_image = diff.detach().cpu().numpy().reshape((s,s,nA))
    ave_image = ave.detach().cpu().numpy().reshape((s,s,nA))

    error = out_image - image

    save_animation(ave_image, time_int, args.out_folder + "Static_component.gif" )
    save_animation(out_image, time_int, args.out_folder + "Reconstructed.gif" )
    save_animation(diff_image, time_int, args.out_folder + "Dynamic_component.gif" )
    save_animation((image_vec - out.cpu().detach().numpy()).reshape((s,s,nA))  , time_int, args.out_folder + "Error.gif" )
    
    Cn = dC.cpu().detach().numpy()
    print(Cn.shape)
    plt.clf()
    plt.imshow(np.abs(Cn))
    plt.colorbar()
    plt.savefig(args.out_folder + "dynamic_coeff_magnitudes.png")
