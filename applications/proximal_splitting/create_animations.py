from time import time
import torch


import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

import sys
import argparse
sys.path.append( "../../")
from nfCRT import *

def save_animation(image, time_int, filename):
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', '5%', '5%')
    im = ax.imshow(image[:,:,0], cmap = 'gray', origin = 'lower')
    #im = ax.imshow(image[:,:,0], origin = 'lower')
    cb = fig.colorbar(im,cax = cax)
    #cb.remove()

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
                        default= 'NF_Prox_Grad/',
                        type = str)
    args = parser.parse_args()
    if torch.cuda.is_available():
        print("Cuda available, Implementing with GPU")
        dev = torch.device('cuda:0')
    else:
        print("Cuda not available, Implementing with CPU")
        dev = torch.device('cpu')

    image = np.load('../initial_pressure.npy')
    s = image.shape[0]
    nA = image.shape[2]
    
    time_int = torch.linspace(-1/2, 1/2,nA+1)
    time_int = time_int[:nA] + (time_int[1] -time_int[0])/2
    x_int = torch.linspace(-1/2,1/2,s+1) 
    x_int = x_int[:s]+  (x_int[1] -x_int[0])/2
    x1, x0, t =torch.meshgrid(x_int, x_int, time_int)
    x0 = x0.flatten()
    x1 = x1.flatten()
    t = t.flatten()
    x = torch.stack((x0, x1, t))
    x = torch.transpose(x,0,1).float().to(dev)

    #save_animation(image, time_int, args.out_folder + 'Phantom.gif')
    image_vec = image.flatten()
    net = torch.load(args.out_folder + 'dynamic_net.pt')
    dpou = net['pou']
    dC = net['C']
    
    
    snet = torch.load(args.out_folder + 'static_net.pt')
    spou = snet['pou']
    sC = snet['C']
    

    mx = 3
    mt = 1
    geodim = 3
    P = Polynomial(geodim, mx, mt = mt, st_separability = True )
    sP = Polynomial(geodim-1, mx, mt = mt, st_separability = False )
    
    
    

    out = torch.zeros(x.size()[0]).to(dev)
    static = torch.zeros(x.size()[0]).to(dev)
    
    
    r = int(x.size()[0]/nA)
    for i in range(nA):
        y = x[r*i:r*(i+1),:]
        out[r*i:r*(i+1)] = torch.einsum( 'ij,jk,ki -> i', dpou(y)[:,:-1], dC, P(y)).detach()
        static[r*i:r*(i+1)] = torch.einsum( 'ij,jk,ki -> i', spou(y[:,0:-1])[:,:-1], sC, sP(y[:,0:-1])).detach()
        


    
    
    print("RMSE = ", np.linalg.norm(image_vec - out.cpu().detach().numpy()  )/np.linalg.norm(image_vec))
    print("Error = ", np.linalg.norm(image_vec - out.cpu().detach().numpy()  ))
    print("Norm of Image = ", np.linalg.norm(image_vec))
    print("Static Baseline = ", np.linalg.norm(image_vec - static.cpu().detach().numpy()  )/np.linalg.norm(image_vec))
    out_image = out.detach().cpu().numpy().reshape((s,s,nA))
    static_image = static.detach().cpu().numpy().reshape((s,s,nA))
    
    

    error = out_image - image
    #save_animation(error, time_int, args.out_folder + "Error.gif" )
    save_animation(out_image, time_int, args.out_folder + "Reconstructed.gif" )
    save_animation(static_image, time_int, args.out_folder + "Static.gif" )
    