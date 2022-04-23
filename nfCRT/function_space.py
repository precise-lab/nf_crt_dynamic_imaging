import numpy as np
import math
import torch
"""
Functions and Classes defining different types of spanning polynomial
"""


def poly(x, m):
    """
    Function that returns polynomial basis of degree 'm', evaluated at points 'x'
    """
    geodim = x.size()[1]
    
    if geodim == 0:
        return None
    if geodim == 1:
        P = torch.zeros(m+1, x.size()[0]).to(x.get_device())
        for i in range(m+1):
            P[i,:] = torch.flatten(x**i)
        return P
    else:
        N = int(math.factorial(m + geodim)/(math.factorial(geodim)*math.factorial(m)))
        P = torch.zeros(N,x.size()[0]).to(x.get_device())

        R = 0
        for i in range(m+1):
            P1 = poly(x[:,1:], m-i)
            r = P1.size()[0]
            P[R:R+r, :] = torch.flatten(x[:,0]**i)*P1
            R += r
        return P

def st_poly(x,m, mt):
    """
    Function that returns polynomial basis for separable degrees in space and time, evaluated at points 'x'
        - 'm' is the degree of the polynomial in space
        - 'mt' is the degree if the polynomial in time
    """
    geodim = x.size()[1]
    if geodim == 0:
        return None
    if geodim == 1:
        P = torch.zeros(mt+1, x.size()[0]).to(x.get_device())
        for i in range(mt+1):
            P[i,:] = torch.flatten(x**i)
        return P
    else:
        Nx = int(math.factorial(m + geodim - 1)/(math.factorial(geodim - 1)*math.factorial(m)))
        Nt = mt + 1
        N = Nx*Nt
        P = torch.zeros(N,x.size()[0]).to(x.get_device())

        R = 0
        for i in range(m+1):
            P1 = st_poly(x[:,1:], m-i, mt)
            r = P1.size()[0]
            P[R:R+r, :] = torch.flatten(x[:,0]**i)*P1
            R += r
        return P

class Polynomial:
    """
    Class that describes polynomial basis 

    Attributes:
    - 'geodim':            geometric dimension of domain
    - 'st_separability':   Boolean describing if space and time are sepearable
    - 'm':                 combined degree of polynomial if 'st_separability' is False or degree of polynomial in space
    - 'mt':                degree of Polynomial in time if 'st_separability' is True
    - 'N':                 number of polynomial basis functions
    """
    def __init__(self, geodim, m, mt = 0, st_separability = False ):
        self.m = m
        self.mt = mt
        self.geodim = geodim
        self.st_separability = st_separability
        if self.st_separability:
            Nx = int(math.factorial(m + geodim - 1)/(math.factorial(geodim - 1)*math.factorial(m)))
            Nt = mt + 1
            self.N = Nx*Nt
        else:
            self.N = int(math.factorial(m + geodim)/(math.factorial(geodim)*math.factorial(m)))
    def __call__(self, x):
        assert x.size()[1] == self.geodim
        if self.st_separability:
            return st_poly(x, self.m, self.mt)
        else:
            return poly(x, self.m)

class PolynomialTP:
    """
    Class that describes polynomial basis 

    Attributes:
    - 'geodim':            geometric dimension of domain
    - 'st_separability':   Boolean describing if space and time are sepearable
    - 'm':                 combined degree of polynomial if 'st_separability' is False or degree of polynomial in space
    - 'mt':                degree of Polynomial in time if 'st_separability' is True
    - 'N':                 number of polynomial basis functions
    """
    def __init__(self, geodim, m, mt):
        self.m = m
        self.mt = mt
        self.geodim = geodim
    
        Nx = (m+1)**(geodim - 1)
        Nt = mt + 1
        self.N = Nx*Nt
    def __call__(self, x):
        assert x.size()[1] == self.geodim
        P = torch.ones(1,x.size(0)).to(x.get_device())
        for i in range(self.geodim - 1):
            P1 = torch.zeros(self.m+1,x.size(0)).to(x.get_device())
            for j in range(self.m+1):
                P1[j,:] = torch.flatten(x[:,i]**j)
            P = torch.einsum('ij, mj -> ijm',P, P1).reshape((self.m+1)**(i+1), x.size(0)).to(x.get_device())
        P1 = torch.zeros(self.mt+1, x.size(0)).to(x.get_device())
        for j in range(self.mt+1):
            P1[j,:] = torch.flatten(x[:,-1]**j)
        P = torch.einsum('ij, mj -> ijm',P, P1).reshape(self.N, x.size(0)).to(x.get_device())
        return P

def sin_basis_size(m, d):
    """
    Function that calculates the size of a sinusoidal basis of degree m
    """
    if d == 1:
        return 2*m+1
    else:
        n = 0
        for i in range(-m, m+1):
            n += sin_basis_size(m-abs(i),d-1)
        return

def sinusoidal(x,m, L):
    """
    Function that returns sinusoidal basis of degree 'm', period 'L', evaluated at points 'x'
    """
    geodim = x.size()[1]
    if geodim == 0:
        return None
    if geodim == 1:
        P = torch.zeros(2*m+1, x.size()[0])
        P[0,:] = 1
        for i in range(1, m+1):
            P[2*i-1,:] = torch.flatten(torch.sin(2*np.pi*i*x/L))
            P[2*i,:] = torch.flatten(torch.sin(2*np.pi*i*x/L))
        return P
    else:
        N = sin_basis_size(m,geodim)
        P = torch.zeros(N,x.size()[0]).to(x.get_device())
    
        P1 = sinusoidal(x[1:0], m, L)
        r = P1.size()[0]
        P[0:r, :] = P1
        R = r

        for i in range(m+1):
            P1 = sinusoidal(x[1:0], m-i, L)
            r = P1.size()[0]
            P[R:R+r, :] = torch.flatten(torch.sin(2*np.pi*i*x/L))*P1
            R += r
            P[R:R+r, :] = torch.flatten(torch.cos(2*np.pi*i*x/L))*P1
            R += r
        return P

class Sinusoidal:
    """
    Class that describes sinusoidal basis 

    Attributes:
    - 'geodim':            geometric dimension of domain
    - 'm':                 combined degree of sinusoidal functions in space and time
    - 'N':                 number of polynomial basis functions
    """
    def __init__(self, geodim, m, L):
        self.geodim = geodim
        self.m = m
        self.L = L
        self.N = sin_basis_size(m,geodim)

    def __call__(self, x):
        assert x.size()[1] == self.geodim
        return sinusoidal(x, self.m, self.L)

    
