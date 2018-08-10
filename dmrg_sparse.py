#! /usr/bin/env python 

"""
Simple implementation of DMRG.
Written by:
    Zhihao Cui
    Garnet Chan
"""

"""
Current convention for MPS and MPO indexing.

    MPS:
          n     L _ _ R
        l_|_r ,    |
                   N
        stored as left-mid-right, e.g. 'lnr', 'LNR'.

    MPO:  
          N
        a_|_b
          |
          n
        stored as 'aNnb'.
    
    Opr:
         _L 
        |_a 
        |_l
        stored as 'Lal'.
        convention:
            from up to down.

    Intermediate result:
         _L 
        |_a n
        |___|___l
        
        stored as 'Lanl'.
        convention:
            first left to right
            then up to down. (anti-clockwise)
            for computational efficiency, the intermediate 
            can be stored not according to such principle,
            but the final result should be.

"""

import sys
from enum import Enum
import numpy as np
from numpy import einsum, reshape, diag
import linalg
import MPSblas
from pyscf import lib
import sparse
from sparse import BCOO
from sparse.butils import assert_eq
from sparse.bcoo import bcalc



class Arrow(Enum):
    """
    Arrow class to indicate the direction of sweep.
    
    0 for left and 1 for right.

    """

    LeftArrow = 0
    RightArrow = 1

def diag_onesite(mpo0, lopr, ropr):
    """
    Compute the diagonal elements of sandwich <L|MPO|R>,
    used as preconditioner of Davidson algorithm.

    Math
    ----------

     _l n r_
    |___|___|
    |_l | r_|
        n

    Parameters
    ----------
    mpo0 : ndarray
        The MPO.
    lopr : ndarray
        left block operators
    ropr : ndarray
        right block operators
     
    Returns
    -------
    diag : ndarray
        The diagonal element, stored as 'lnr'.

    """

    mpo0_diag = einsum('annb -> anb', mpo0)
    lopr_diag = einsum('lal -> la', lopr)
    ropr_diag = einsum('rbr -> br', ropr)

    scr1 = einsum('la, anb -> lnb', lopr_diag, mpo0_diag)
    diag = einsum('lnb, br -> lnr', scr1, ropr_diag)

    #diag = scr2
    # ZHC NOTE check the SDcopy of upcast options
    return diag

def diag_twosite(lmpo, rmpo, lopr, ropr):
    lmpo_diag = einsum('annb -> anb', lmpo)
    rmpo_diag = einsum('bmmc -> bmc', rmpo)
    lopr = einsum('lal->la', lopr)
    ropr = einsum('rcr-> cr', ropr)

    scr1 = einsum('la, anb -> lnb', lopr, lmpo)
    scr2 = einsum('lnb, bmc -> lnmc', scr1, rmpo)
    diag = einsum('lnmc, cr -> lnmr', scr2, ropr)
    return diag

def dot_onesite(mpo0, lopr, ropr, wfn0):
    """
    Compute the sigma vector, i.e. sigma = H * c
    used for Davidson algorithm.

    Math
    ----------

     _L N R_
    |___|___|
    |___|___|


    Parameters
    ----------
    mpo0 : ndarray
        The MPO.
    lopr : ndarray
        left block operators
    ropr : ndarray
        right block operators
    wfn0 : ndarray
        The current MPS. (wavefunction for desired roots)
     
    Returns
    -------
    sgv0 : ndarray
        The sigma vector, stored as LNR.

    """
    # ZHC NOTE the contraction order and stored structure may be optimized.

    scr1 = einsum('Lal, lnr -> Lanr', lopr, wfn0)
    scr2 = einsum('Lanr, aNnb -> LNbr', scr1, mpo0)
    sgv0 = einsum('LNbr, Rbr -> LNR', scr2, ropr)
    return sgv0

def dot_twosite(lmpo, rmpo, lopr, ropr, wfn0):
    """
    Compute the sigma vector, i.e. sigma = H * c
    used for Davidson algorithm, in the twosite algorithm

     _L N M R_
    |___|_|___|
    |___|_|___|
    """
    scr1 = einsum("Lal, lnmr -> Lanmr", lopr, wfn0)
    scr2 = einsum("Lanmr, aNnb -> LNbmr", scr1, lmpo)
    scr3 = einsum("LNbmr, bMmc -> LNMcr", scr2, rmpo)
    sgv0 = einsum("LNMcr, Rcr -> LNMR", scr3, ropr)
    return sgv0
	

def canonicalize(forward, wfn0, M = 0):
    """
    Canonicalize the wavefunction.
    
    Parameters
    ----------
    forward : int 
        0 for left and 1 for right.
    wfn0 : ndarray
        current MPS.
    M : int
        bond dimension
     
    Returns
    -------
    mps0 : ndarray
        canonicalized mps.
    gaug : ndarray
        gauge, i.e. sigma * v

    """

    if forward:
        mps0, s, wfn1, dwt = linalg.svd("ij, k", wfn0, M)
        gaug = einsum("ij, jk -> ik", diag(s), wfn1)
    else:
        wfn1, s, mps0, dwt = linalg.svd("i, jk", wfn0, M)
        gaug = einsum("ij, jk -> ik", wfn1, diag(s))
    return mps0, gaug
        

def renormalize(forward, mpo0, opr0, bra0, ket0):
    """
    Renormalized the block opr.
    
    Parameters
    ----------
    forward : int 
        0 for left and 1 for right.
    mpo0 : ndarray
        MPO.
    opr0 : ndarray
        block opr.
    bra0 : ndarray
        upper MPS. should already be conjugated.
    ket0 : ndarray
        down MPS
     
    Returns
    -------
    opr1 : ndarray
        renormalized block opr.

    """
 
    if forward:
        scr = einsum('Lal, lnr -> Lanr', opr0, ket0)
        scr = einsum('Lanr, aNnb -> LNbr', scr, mpo0)
        opr1 = einsum('LNR, LNbr -> Rbr', bra0, scr)
    else:
        scr = einsum('LNR, Rbr -> LNbr', bra0, opr0)
        scr = einsum('LNbr, aNnb -> Lanr', scr, mpo0)
        opr1 = einsum('Lanr, lnr-> Lal ', scr, ket0)
    
    return opr1    

def eig_onesite(forward, mpo0, lopr, ropr, wfn0, M, tol, nroots=1):
     diag_flat = diag_onesite(mpo0, lopr, ropr).ravel()
     mps_shape = wfn0.shape
     
     def dot_flat(x):
         return dot_onesite(mpo0, lopr, ropr, x.reshape(mps_shape)).ravel()
     def compute_precond_flat(dx, e, x0):
         return dx / (diag_flat - e)

     energy, wfn0s = lib.linalg_helper.davidson(dot_flat, wfn0.ravel(),
                                                compute_precond_flat, tol = tol, nroots = nroots)

     wfn0s = [wfn0.reshape(mps_shape) for wfn0 in wfn0s]

     # implement state average ...
     wfn0, gaug = canonicalize(forward, wfn0, M) # wfn0 becomes left/right canonical
     return wfn0, gaug


def optimize_onesite(forward, mpo0, lopr, ropr, wfn0, wfn1, M, tol):
    """
    Optimization for onesite algorithm.
    
    Parameters
    ----------
    forward : int 
        0 for left and 1 for right.
    mpo0 : ndarray
        MPO.
    lopr : ndarray
        left block opr.
    ropr : ndarray
        right block opr.
    wfn0 : ndarray
        MPS for canonicalization.
    wfn1 : ndarray
        MPS.
    M : int
        bond dimension
     
    Returns
    -------
    energy : float or list of floats
        The energy of desired root(s).

    """

    diag_flat = diag_onesite(mpo0, lopr, ropr).ravel()
    
    mps_shape = wfn0.shape
    def dot_flat(x):
        return dot_onesite(mpo0, lopr, ropr, x.reshape(mps_shape)).ravel()
    def compute_precond_flat(dx, e, x0):
        return dx / (diag_flat - e)


    energy, wfn0 = lib.linalg_helper.davidson(dot_flat, wfn0.ravel(), compute_precond_flat, tol = tol)
    wfn0 = wfn0.reshape(mps_shape)
    
    if forward:
        wfn0, gaug = canonicalize(1, wfn0, M) # wfn0 R => lmps gaug
        wfn1 = einsum("ij,jkl->ikl", gaug, wfn1)
        lopr = renormalize(1, mpo0, lopr, wfn0.conj(), wfn0)
        return energy, wfn0, wfn1, lopr
    else:
        wfn0, gaug = canonicalize(0, wfn0, M) # wfn0 R => lmps gaug
        wfn1 = einsum("ijk,kl->ijl", wfn1, gaug)
        ropr = renormalize(0, mpo0, ropr, wfn0.conj(), wfn0)
        return energy, wfn0, wfn1, ropr

def optimize_twosite(forward, lmpo, rmpo, lopr, ropr, lwfn, rwfn, M, tol):
    """
    Optimization for twosite algorithm.
    
    Parameters
    ----------
    M : int
        bond dimension
     
    Returns
    -------
    energy : float or list of floats
        The energy of desired root(s).

    """
    wfn2 = einsum("lnr, rms -> lnms", lwfn, rwfn)
    diag = diag_twosite(lmpo, rmpo, lopr, ropr)

    mps_shape = wfn2.shape
    
    def dot_flat(x):
        return dot_twosite(lmpo, rmpo, lopr, ropr, x.reshape(mps_shape)).ravel()
    def compute_precond_flat(dx, e, x0):
        return dx / (diag_flat - e)

    energy, wfn0 = lib.linalg_helper.davidson(dot_flat, wfn2.ravel(), compute_precond_flat)
    wfn0 = wfn0.reshape(mps_shape)

    if forward:
        wfn0, gaug = canonicalize(1, wfn0, M) # wfn0 R => lmps gaug
        wfn1 = einsum("ij, jkl -> ikl", gaug, wfn1)
        lopr = renormalize(1, mpo0, lopr, wfn0.conj(), wfn0)
        return energy, wfn0, wfn1, lopr
    else:
        wfn0, gaug = canonicalize(0, wfn0, M) # wfn0 R => lmps gaug
        wfn1 = einsum("ijk, kl -> ijl", wfn1, gaug)
        ropr = renormalize(0, mpo0, ropr, wfn0.conj(), wfn0)
        return energy, wfn0, wfn1, ropr




def sweep(mpos, mpss, loprs, roprs, algo = 'onsite', M = 1, tol = 1e-6):
    emin = 1.0e8
    print "\t++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    print "\t\t\tFORWARD SWEEP"
    print "\t++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    
    N = len(mpss)
    assert(len(mpos) == N);

    for i in xrange(0, N - 1): 

        print "\t===================================================================================================="
        print "\t\tSITE [  %5d  ] "%i
        print "\t----------------------------------------------------------------------------------------------------"

        # ZHC NOTE : in future the mps, mpo and operator should be read from somewhere!
        #print "\t\tloading operators and wavefunction of next site (env)..."
        #load(mpos[i+1], get_mpofile(input.prefix, i+1));
        #load(mpss[i+1], get_mpsfile(input.prefix, RIGHTCANONICAL, i+1));
        #cout << "done" << endl;
        
        sys.stdout.flush()

        # diagonalize
        if(algo == 'onesite'):
            print "\t\toptimizing wavefunction: 1-site algorithm "
            #load(ropr, get_oprfile(input.prefix, RIGHTCANONICAL, i))
            # ZHC NOTE store the old operators and mps
            # ZHC NOTE we should at the beginning allocate a buffer which can handle all oprs!
            ropr = roprs.pop()
            eswp, mpss[i], mpss[i + 1], lopr = optimize_onesite(1, mpos[i], loprs[-1], ropr, mpss[i], mpss[i + 1], M, 0.1 * tol)
            loprs.append(lopr)

        else:
            print "\t\toptimizing wavefunction: 2-site algorithm "
            #load(ropr, get_oprfile(input.prefix, RIGHTCANONICAL, i+1));
            eswp, wfn0, wfn1, lopr, ropr = optimize_twosite(1, mpos[i], mpos[i + 1], lopr, ropr, mpss[i], mpss[i + 1], M, 0.1 * tol)
            #eswp = optimize_twosite_merged(1, mpos[i], mpos[i+1], lopr, ropr, mpss[i], mpss[i+1], 0.1*T, M);
        

        if(eswp < emin):
            emin = eswp

        # print result
        print "\t\t--------------------------------------------------------------------------------"
        print "\t\t\tEnergy = %20.10f "%eswp
        print "\t\t--------------------------------------------------------------------------------"

        #print "\t\tsaving operators and wavefunction of this site (sys)..."
        sys.stdout.flush()
        #save(mpss[i], get_mpsfile(input.prefix, LEFTCANONICAL, i));
        #save(lopr,    get_oprfile(input.prefix, LEFTCANONICAL, i+1));
        #mpos[i].clear(); ZHC NOTE
        #mpss[i].clear();
        print "done"
    #save(mpss[N-1], get_mpsfile(input.prefix, WAVEFUNCTION, N-1));

    print "\t++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" 
    print "\t\t\tBACKWARD SWEEP" 
    print "\t++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" 

    #load(ropr, get_oprfile(input.prefix, RIGHTCANONICAL, N-1));

    for i in xrange(N - 1, 0, -1):
          
        print "\t===================================================================================================="
        print "\t\tSITE [  %5d  ] "%i
        print "\t----------------------------------------------------------------------------------------------------"

        #print "\t\tloading operators and wavefunction of next site (env)..."
        #load(mpos[i-1], get_mpofile(input.prefix, i-1));
        #load(mpss[i-1], get_mpsfile(input.prefix, LEFTCANONICAL, i-1));
        #print "done"

        # diagonalize
        if(algo == 'onesite'):
            print "\t\toptimizing wavefunction: 1-site algorithm "
            #load(ropr, get_oprfile(input.prefix, RIGHTCANONICAL, i))
            
            lopr = loprs.pop()
            eswp, mpss[i], mpss[i - 1], ropr = optimize_onesite(0, mpos[i], lopr, roprs[-1], mpss[i], mpss[i - 1], M, 0.1 * tol)
            roprs.append(ropr)

        else:
            print "\t\toptimizing wavefunction: 2-site algorithm "
            #load(ropr, get_oprfile(input.prefix, RIGHTCANONICAL, i+1));
            eswp = optimize_twosite(0, mpos[i - 1], mpos[i], lopr, ropr, mpss[i - 1], mpss[i], 0.1 * tol, M)
            #eswp = optimize_twosite_merged(0, mpos[i - 1], mpos[i], lopr, ropr, mpss[i - 1], mpss[i], 0.1*T, M);
        
        if(eswp < emin):
            emin = eswp
        # print result
        print "\t\t--------------------------------------------------------------------------------"
        print "\t\t\tEnergy = %20.10f "%eswp
        print "\t\t--------------------------------------------------------------------------------"

        #print "\t\tsaving operators and wavefunction for this site (sys)..." << flush;
        #save(mpss[i], get_mpsfile(input.prefix, RIGHTCANONICAL, i));
        #save(ropr,    get_oprfile(input.prefix, RIGHTCANONICAL, i-1));
        #mpos[i].clear();
        #mpss[i].clear();
        print "done"
    #save(mpss[0], get_mpsfile(input.prefix, WAVEFUNCTION, 0));
    #mpos[0].clear();
    #mpss[0].clear();

    print "\t===================================================================================================="

    return emin
    


def heisenberg_mpo_dense(N, h, J):
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


def heisenberg_mpo_sparse(N, h, J):
    """
    Create Heisenberg MPO, in BCOO format.
    """
    Sp = np.array([[0.0, 1.0], [0.0, 0.0]])
    Sm = np.array([[0.0, 0.0], [1.0, 0.0]])
    Sz = np.array([[0.5, 0.0], [0.0, -0.5]])
    I = np.array([[1.0, 0.0], [0.0, 1.0]])
    z = np.array([[0.0, 0.0], [0.0, 0.0]])
    
    W = []
    scr_data = np.array([[[-h * Sz]], [[0.5 * J * Sm]], [[0.5 * J * Sp]], [[J * Sz]], [[I]]])
    scr_coords = np.array([[0, 0, 0, 0, 0],
                           [0, 1, 2, 3, 4],
                           [0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0]]) 
    scr_bcoo = BCOO(scr_coords, data = scr_data, shape = (1, 5, 2, 2), block_shape = (1, 1, 2, 2))
    scr_bcoo = scr_bcoo.transpose((0, 3, 2, 1))
    W.append(scr_bcoo)
    for i in xrange(N-2):
        scr_data = np.array([[[I]],
                        [[Sp]],
                        [[Sm]],
                        [[Sz]],
                        [[-h * Sz]], [[0.5 * J * Sm]], [[0.5 * J * Sp]], [[J * Sz]], [[I]]])
        scr_coords = np.array([[0, 1, 2, 3, 4, 4, 4, 4, 4],
                               [0, 0, 0, 0, 0, 1, 2, 3, 4],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0]]) 
        scr_bcoo = BCOO(scr_coords, data = scr_data, shape = (5, 5, 2, 2), block_shape = (1, 1, 2, 2))
        scr_bcoo = scr_bcoo.transpose((0, 3, 2, 1))
        W.append(scr_bcoo)
        
    scr_data = np.array([[[I]], [[Sp]], [[Sm]], [[Sz]], [[-h * Sz]]])
    scr_coords = np.array([[0, 1, 2, 3, 4],
                           [0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0]]) 
    scr_bcoo = BCOO(scr_coords, data = scr_data, shape = (5, 1, 2, 2), block_shape = (1, 1, 2, 2))
    scr_bcoo = scr_bcoo.transpose((0, 3, 2, 1))
    W.append(scr_bcoo)
    return W


def test_eq_dense_sparse(mpxs_dense, mpxs_sparse):
    assert(len(mpxs_dense) == len(mpxs_sparse))
    for i in xrange(len(mpxs_dense)):
        print i
        assert_eq(mpxs_dense[i], mpxs_sparse[i])
        print "ok"
    return True

def initialize_heisenberg(N, h, J, M):
    """
    Initialize the MPS, MPO, lopr and ropr.
    """
    # MPS
    mpss = MPSblas.rand([2] * N, D = M, seed = 0) 
    normalize_factor = 1.0 / MPSblas.norm(mpss) 
    mpss = MPSblas.mul(normalize_factor, mpss) 
  
    # make MPS right canonical
    for i in xrange(N - 1, 0, -1):
        mpss[i], gaug = canonicalize(0, mpss[i], M = M)
        mpss[i - 1] = einsum("ijk, kl -> ijl", mpss[i - 1], gaug)

    # MPO
    mpos_dense = np.asarray(heisenberg_mpo_dense(N, h, J))
    mpos_sparse = heisenberg_mpo_sparse(N, h, J)
    
    test_eq_dense_sparse(mpos_dense, mpos_sparse)
    exit() 

    # lopr
    loprs = [np.array([[[1.0]]])]
    # ropr
    roprs = [np.array([[[1.0]]])]
    for i in xrange(N - 1, 0, -1):
        roprs.append(renormalize(0, mpos[i], roprs[-1], mpss[i].conj(), mpss[i]))
    
    # NOTE the loprs and roprs should be list currently to support pop()!
    return mpss, mpos, loprs, roprs
    
def test():
    N = 5
    h = 1.0
    J = 1.0
    M = 50
    mpss, mpos, loprs, roprs = initialize_heisenberg(N, h, J, M)    
    print mpos
    for i in xrange(len(mpos)):
        print mpos[i].shape

    exit()
    
    energy = sweep(mpos, mpss, loprs, roprs, algo = 'onesite', M = M, tol = 1e-6)
    
    print "Energy :", energy
    print "Energy per site: ", energy/float(N)
    
if __name__ == '__main__':
    test()


