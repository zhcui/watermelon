import sys
from enum import Enum
import numpy as np
from numpy import einsum, reshape, diag
import linalg
import MPSblas
from pyscf import lib

def heisenberg_mpo(N, h, J):
    """
    Create Heisenberg MPO.
    """
    Sp = np.array([[0.0, 1.0], [0.0, 0.0]])
    Sm = np.array([[0.0, 0.0], [1.0, 0.0]])
    Sz = np.array([[0.5, 0.0], [0.0, -0.5]])
    I = np.array([[1.0, 0.0], [0.0, 1.0]])
    z = np.array([[0.0, 0.0], [0.0, 0.0]])
    
    W = []
    W.append(einsum('abnN -> aNnb', np.array([[-h * Sz, 0.5 * J * Sm, 0.5 * J * Sp, J * Sz, I]])))
    for i in xrange(N-2):
        W.append(einsum('abnN -> aNnb', np.array([[I, z, z, z, z],
                                                    [Sp, z, z, z, z],
                                                    [Sm, z, z, z, z],
                                                    [Sz, z, z, z, z],
                                                    [-h * Sz, 0.5 * J * Sm, 0.5 * J * Sp, J * Sz, I]])))
    W.append(einsum('abnN -> aNnb', np.array([[I], [Sp], [Sm], [Sz], [-h * Sz]])))
    return W


def holstein_mpo(n_bath, d_boson, e0, eks, vks):
    """
    Create Holstein/spin-boson model with one spin (at the left-most site)

    n_bath:   number of bath sites
    d_boson:  physical bond dim of bosonic sites
    e0:       energy of the spin 
    eks:      energy of boson k (1, ..., n_bath)
    vks:      coupling of spin to boson k
              ohmic spectral density: vks = sqrt(2*alpha*omega*exp(omega/omega_c))
    """

    bN = np.diag(range(d_boson))
    bp = np.diag([np.sqrt(x) for x in range(1,d_boson)],k=-1)  # -1 below diagonal
    bm = np.diag([np.sqrt(x) for x in range(1,d_boson)],k= 1)  #  1 above diagonal 
    Sx = np.array([[0.,1.],[1.,0.]])
    Sz = np.array([[1.,0.],[0.,-1.]])

    Dw = 3
    df = 2
    db = d_boson
    phys_bonds = [df] + [db]*n_bath

    L = n_bath + 1
    
    W = []

    ## MPO on spin site
    WS = np.zeros([1,df,df,self.Dw])
    WS[0,:,:,0] = e0*self.ops['SX']              # site energy
    WS[0,:,:,1] = self.ops['SZ']                    # creation of fermion
    WS[0,:,:,2] = np.eye(df)
    W.append(Ws)

    ## MPO on bath sites
    Wb = np.zeros([self.Dw,db,db,self.Dw])
    Wb[0,:,:,0] = np.eye(db)
    Wb[1,:,:,1] = np.eye(db)                 # propagates SZ interaction term
    Wb[2,:,:,2] = np.eye(db)

    for i in range(L-1):            
        Wb[1,:,:,0] = vks[i]*self.ops['b-'] + np.conj(vks[i])*self.ops['b+'] # interaction with spin
        Wb[2,:,:,0] = eks[i]*self.ops['bn']  # energy of boson 
        W.append(Wb)
        

    WL = np.zeros([self.Dw,db,db,1])
    WL[0,:,:,0] = np.eye(db)
    WL[1,:,:,0] = vks[-1]*self.ops['b-'] + np.conj(vks[-1])*self.ops['b+'] # interaction with spin
    WL[2,:,:,0] = eks[-1]*self.ops['bn']    # energy of boson
    W.append(WL)

    return W
