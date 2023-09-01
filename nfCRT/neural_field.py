import torch
class NeuralField:
    def __init__(self, psi, P, dev, background = False):
        self.psi = psi
        self.background = background
        if not background:
            try:
                self.C = torch.nn.Parameter(torch.zeros((psi.N_part, P.N), requires_grad = True, device = dev))
            except:
                self.C = torch.nn.Parameter(torch.zeros((psi.N_part, P.N), requires_grad = True))
        else:
            try:
                self.C = torch.nn.Parameter(torch.zeros((psi.N_part-1, P.N), requires_grad = True, device = dev))
            except:
                self.C = torch.nn.Parameter(torch.zeros((psi.N_part-1, P.N), requires_grad = True))
        self.P = P
    def __call__(self, x):
        if not self.background:
            return torch.einsum( 'ij,jk,ki -> i', self.psi(x), self.C, self.P(x))
        else:
            return torch.einsum( 'ij,jk,ki -> i', self.psi(x)[:,:-1], self.C, self.P(x))
    def copy(self):
        old_psi = copy.deepcopy(self.psi)
        NF2 = NeuralField(old_psi, self.P, self.C.get_device(), background = self.background)
        NF2.C = self.C.detach()
        return NF2 
class NeuralFieldSum:
    def __init__(self, A,B, a, b):
        self.A = A
        self.B = B
        self.a = a
        self.b = b
    def __call__(self, x):
        if self.b == 0:
            return self.a*self.A(x)
        if isinstance(self.B, NeuralField) or isinstance(self.B, NeuralFieldSum) or isinstance(self.B, FrameBasedFunction):
            return self.a*self.A(x) + self.b*self.B(x)
        else:
            print("Invalid Type for Neural Field Sum")
            assert False

class PwNeuralField:
    def __init__(self, psi, C):
        self.psi = psi
        self.C = C
    def __call__(self, x, inds):
        return torch.einsum( 'ij,ij -> i', self.psi(x), self.C[inds,:])
