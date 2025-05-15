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
import torch
import copy
"""Classes implementing partitions of unity and neural field classes
"""

class POU_ResNet(torch.nn.Module):
    def __init__(self, Width, Final_width, geodim = 2, Depth = 3):
        super(POU_ResNet, self).__init__()
        self.Width = Width
        self.Depth = Depth
        self.relu = torch.nn.ReLU()
        self.soft_max = torch.nn.Softmax(dim = 1)
        self.input = torch.nn.Linear(geodim,self.Width)
        self.res_layers= torch.nn.ModuleList([torch.nn.Linear(self.Width, self.Width) for i in range(Depth)])
        self.Fw = Final_width
        self.soft_inp = torch.nn.Linear(self.Width, Final_width, bias = False)
    def init_weights(self):
        with torch.no_grad():
            w = self.Width**(-1/2)
            for l in self.res_layers:
                l.weight.uniform_(-w,w)
                l.bias.uniform_(-1,1)
            
            self.soft_inp.weight.uniform_(-1,1)
       
    def forward(self, x):
        y = self.input(x)
        x = self.input(x)
        for l in self.res_layers:
            x = self.relu(l(x)) + y
        x = self.soft_inp(x)*2*self.Fw
        x = -x**2
        x = self.soft_max(x)
        return x
class POU_Siren(torch.nn.Module):
    """
    Class that implements a POU neural network with sinusoidal activation
    
    Attributes:
    - 'Width': Width of network
    - 'Depth': Depth of network
    - 'inenr_layers': list of lienear layers applied in to sinusoidal functions
    - 'soft_max': soft max to create POU properties
    """
    def __init__(self, Width, Final_Width, Depth = 2, geodim = 2,split = False):
        super(POU_Siren, self).__init__()

        self.split = split
        self.N_part = Final_Width
        self.soft_max = torch.nn.Softmax(dim = 1)
        self.geodim = geodim
        self.Depth = Depth
        self.Width = Width

        self.alpha = (self.Width/140)**(1/2)
        self.omega = (self.N_part/40)**(1/2)
        
        self.input = torch.nn.Linear(geodim,Width)
        self.inner_layers = torch.nn.ModuleList([torch.nn.Linear(Width,Width) for i in range(Depth)])
        if split:
            self.static_input = torch.nn.Linear(geodim-1,Width)
            self.static_layers = torch.nn.ModuleList([torch.nn.Linear(Width,Width) for i in range(Depth)])
            self.soft_inp = torch.nn.Linear(2*Width,Final_Width)
        else:
            self.soft_inp = torch.nn.Linear(Width,Final_Width)

        self.init_weights()
    def init_weights(self):
        with torch.no_grad():
            self.soft_inp.bias.uniform_(0,0)
            self.input.bias.normal_(0, 1/np.sqrt(2))
            if self.split:
                self.static_input.bias.normal_(0, 1/np.sqrt(2))
                for layer, static_layer in zip(self.inner_layers,self.static_layers):
                    layer.bias.uniform_(0,0)
                    static_layer.bias.uniform_(0,0)
            else:
                for layer in self.inner_layers:
                    layer.bias.uniform_(0,0)
                
    def forward(self, x):
        if self.split:
            y = x[:,:-1]
            x = torch.sin(2*np.pi*self.input(self.alpha*x))
            y = torch.sin(2*np.pi*self.static_input(self.alpha*y))
            for layer, static_layer in zip(self.inner_layers,self.static_layers):
                x = torch.sin(8*np.pi*layer(x)/self.Depth)
                y = torch.sin(8*np.pi*layer(y)/self.Depth)
            x = torch.cat((x,y),dim = 1)
        else:
            x = torch.sin(2*np.pi*self.input(self.alpha*x))
            for layer in self.inner_layers:
                x = torch.sin(8*np.pi*layer(x)/self.Depth)
        x = self.soft_inp(self.omega*x)
        x = self.soft_max(x)
        return x

class POU_Tanh(torch.nn.Module):
    """
    Class that implements a POU neural network with sinusoidal activation
    
    Attributes:
    - 'Width': Width of network
    - 'Depth': Depth of network
    - 'inenr_layers': list of lienear layers applied in to sinusoidal functions
    - 'soft_max': soft max to create POU properties
    """
    def __init__(self, Width, Final_Width, Depth = 2, geodim = 2):
        super(POU_Tanh, self).__init__()
        self.Width = Width
        self.soft_max = torch.nn.Softmax(dim = 1)
        self.geodim = geodim
        
        self.input = torch.nn.Linear(geodim,self.Width)
        self.inner_layers = torch.nn.ModuleList([torch.nn.Linear(self.Width,self.Width) for i in range(Depth)])
        self.soft_inp = torch.nn.Linear(self.Width,Final_Width)
        #self.init_weights()
    def init_weights(self):
        with torch.no_grad():
            self.soft_inp.bias.uniform_(0,0)
            self.soft_inp.weight.uniform_(0,0)
            self.input.bias.uniform_(0,0)
            self.input.weight.uniform_(0,0)
            for layer in self.inner_layers:
                layer.bias.uniform_(0,0)
                layer.weight.uniform_(0,0)
    def forward(self, x):
        x = torch.tanh(self.input(x))
        for layer in self.inner_layers:
            x = torch.tanh(layer(x))
        x = self.soft_inp(x)
        x = self.soft_max(x)
        return x

        


        
