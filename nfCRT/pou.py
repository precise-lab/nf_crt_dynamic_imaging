
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
    def __init__(self, Width, Final_Width, Depth = 2, geodim = 2):
        super(POU_Siren, self).__init__()
        self.N_part = Final_Width
        self.soft_max = torch.nn.Softmax(dim = 1)
        self.geodim = geodim
        self.Width = Width
        
        self.input = torch.nn.Linear(geodim,Width)
        self.inner_layers = torch.nn.ModuleList([torch.nn.Linear(Width,Width) for i in range(Depth)])
        self.soft_inp = torch.nn.Linear(Width,Final_Width)

        self.init_weights()
    def init_weights(self):
        with torch.no_grad():
            self.soft_inp.bias.uniform_(0,0)
            self.input.bias.normal_(0, 1/np.sqrt(2))
            for layer in self.inner_layers:
                layer.bias.uniform_(0,0)
    def forward(self, x):
        x = torch.sin(2*np.pi*self.input(x))
        for layer in self.inner_layers:
            x = torch.sin(2*np.pi*layer(x))
        x = self.soft_inp(x)
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

class NeuralField:
    def __init__(self, psi, P, dev):
        self.psi = psi
        self.C = torch.nn.Parameter(torch.zeros((psi.N_part, P.N), requires_grad = True, device = dev))
        self.P = P
    def __call__(self, x):
        return torch.einsum( 'ij,jk,ki -> i', self.psi(x), self.C, self.P(x))
    def copy(self):
        old_psi = copy.deepcopy(self.psi)
        NF2 = NeuralField(old_psi, self.P, self.C.get_device())
        NF2.C = self.C.detach()
        return NF2 

class NeuralFieldSum:
    def __init__(self, A,B, a, b):
        self.A = A
        self.B = B
        self.a = a
        self.b = b
    def __call__(self, x):
        if isinstance(self.B, NeuralField) or isinstance(self.B, NeuralFieldSum):
            return self.a*self.A(x) + self.b*self.B(x)
        else:
            s = 200
            nA = 180
            coords = (s*(x[:,0]+1/2).detach().cpu().numpy()).astype(int) + \
                     s*(s*(x[:,1]+1/2).detach().cpu().numpy()).astype(int) + \
                     s**2*(nA*(x[:,2]+1/2).detach().cpu().numpy()).astype(int) 
            B = torch.from_numpy(self.B[coords]).to(x.get_device())
            return self.a*self.A(x) + self.b*B