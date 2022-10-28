import torch
import torch.nn as nn
import torchdiffeq as ode

class GeneDynamics(nn.Module):
    def __init__(self,  A,  b=1, f=1, h=2):
        super(GeneDynamics, self).__init__()
        self.A = A   # Adjacency matrix
        self.b = b
        self.f = f
        self.h = h

    def forward(self, t, x):
        """
        :param t:  time tick
        :param x:  initial value:  is 2d row vector feature, n * dim
        :return: dxi(t)/dt = -b*xi^f + \sum_{j=1}^{N}Aij xj^h / (1 + xj^h)
        If t is not used, then it is autonomous system, only the time difference matters in numerical computing
        """
        if hasattr(self.A, 'is_sparse') and self.A.is_sparse:
            f = -self.b * (x ** self.f) + torch.sparse.mm(self.A, x**self.h / (x**self.h + 1))
        else:
            f = -self.b * (x ** self.f) + torch.mm(self.A, x ** self.h / (x ** self.h + 1))
        return f



class HeatDiffusion(nn.Module):
    # In this code, row vector: y'^T = y^T A^T      textbook format: column vector y' = A y
    def __init__(self,  L,  k=1):
        super(HeatDiffusion, self).__init__()
        self.L = -L  # Diffusion operator
        self.k = k   # heat capacity

    def forward(self, t, x):
        """
        :param t:  time tick
        :param x:  initial value:  is 2d row vector feature, n * dim
        :return: dX(t)/dt = -k * L *X
        If t is not used, then it is autonomous system, only the time difference matters in numerical computing
        """
        if hasattr(self.L, 'is_sparse') and self.L.is_sparse:
            f = torch.sparse.mm(self.L, x)
        else:
            f = torch.mm(self.L, x)
        return self.k * f


class MutualDynamics(nn.Module):
    #  dx/dt = b +
    def __init__(self, A, b=0.1, k=5., c=1., d=1., e=0.9, h=0.1):
        super(MutualDynamics, self).__init__()
        self.A = A   # Adjacency matrix, symmetric
        self.b = b
        self.k = k
        self.c = c
        self.d = d
        self.e = e
        self.h = h

    def forward(self, t, x):
        """
        :param t:  time tick
        :param x:  initial value:  is 2d row vector feature, n * dim
        :return: dxi(t)/dt = bi + xi(1-xi/ki)(xi/ci-1) + \sum_{j=1}^{N}Aij *xi *xj/(di +ei*xi + hi*xj)
        If t is not used, then it is autonomous system, only the time difference matters in numerical computing
        """
        n, d = x.shape
        f = self.b + x * (1 - x/self.k) * (x/self.c - 1)
        if d == 1:
            # one 1 dim can be computed by matrix form
            if hasattr(self.A, 'is_sparse') and self.A.is_sparse:
                outer = torch.sparse.mm(self.A,
                                        torch.mm(x, x.t()) / (self.d + (self.e * x).repeat(1, n) + (self.h * x.t()).repeat(n, 1)))
            else:
                outer = torch.mm(self.A,
                                 torch.mm(x, x.t()) / (
                                             self.d + (self.e * x).repeat(1, n) + (self.h * x.t()).repeat(n, 1)))
            f += torch.diag(outer).view(-1, 1)
        else:
            # high dim feature, slow iteration
            if hasattr(self.A, 'is_sparse') and self.A.is_sparse:
                vindex = self.A._indices().t()
                for k in range(self.A._values().__len__()):
                    i = vindex[k, 0]
                    j = vindex[k, 1]
                    aij = self.A._values()[k]
                    f[i] += aij * (x[i] * x[j]) / (self.d + self.e * x[i] + self.h * x[j])
            else:
                vindex = self.A.nonzero()
                for index in vindex:
                    i = index[0]
                    j = index[1]
                    f[i] += self.A[i, j]*(x[i] * x[j]) / (self.d + self.e * x[i] + self.h * x[j])
        return f

class SisDynamics(nn.Module):
    def __init__(self, A):
        super(SisDynamics, self).__init__()
        self.A = A   # Adjacency matrix

    def forward(self, t, x):
        """
        :param t:  time tick
        :param x:  initial value:  is 2d row vector feature, n * dim
        :return: dxi(t)/dt = -xi+ \sum_{j=1}^{N}Aij (1 - xj)*xi
        If t is not used, then it is autonomous system, only the time difference matters in numerical computing
        """
        f = -x
        if hasattr(self.A, 'is_sparse') and self.A.is_sparse:
#             f = -x + torch.sparse.mm(self.A, x - torch.sparse.mm(x, x.t()))
            outer = torch.sparse.mm(self.A, x - torch.sparse.mm(x,x))
        else:
            outer = torch.mm(self.A, x - torch.mm(x, x.t()))
        f += torch.diag(outer).view(-1, 1)
        return f
