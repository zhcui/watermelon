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
        stored as 'anNb'.
    
    Opr:
         _L 
        |_a 
        |_l
        stored as 'laL'.
        convention:
            from down to up.

    Intermediate result:
         _L 
        |_a n
        |___|___l
        
        stored as 'lnaL'.
        convention:
            first from down to up,
            then left to right.

"""

import numpy as np
from np import einsum
from pyscf import lib
from enum import Enum

class Arrow(Enum):
    """
    Arrow class to indicate the direction of sweep.
    
    0 for left and 1 for right.

    """

    LeftArrow = 0
    RightArrow = 1

def compute_diagonal_elements(mpo0, lopr, ropr):
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
    mpo0_diag = einsum('rbr -> rb', ropr)

    scr1 = einsum('la, anb -> lnb', lopr_diag, mpo0_diag)
    scr2 = einsum('lnb, rb -> lnr', scr1, ropr_diag)

    diag = scr2
    # ZHC NOTE check the SDcopy of upcast options
    return diag

def compute_diagonal_elements_twosite(lmpo, rmpo, lopr, ropr):
    lmpo_diag = einsum('annb -> anb', lmpo)
    rmpo_diag = einsum('bmmc -> bmc', rmpo)
    lopr = einsum('lal->la', lopr)
    ropr = einsum('rcr->rc', ropr)

    scr1 = einsum('la, anb->lnb', lopr, lmpo)
    scr2 = einsum('lnb, bmc->lnmc', scr1, rmpo)
    scr3 = einsum('lnmc, rc -> lnmr', scr2, ropr)
    return scr3

def compute_sigmavector_twosite(lmpo, rmpo, lopr, ropr, wfn0):
    """
    Compute the sigma vector, i.e. sigma = H * c
    used for Davidson algorithm, in the twosite algorithm

     _L N M R_
    |___|_|___|
    |___|_|___|
    """
    scr1 = einsum("laL,lnmr->aLnmr", lopr, wfn0)
    scr2 = einsum("aLnmr,anNb->LmNbr", scr1, lmpo)
    scr3 = einsum("LmNbr,bmMc->LNMrc", scr2, rmpo)
    sgv0 = einsum("LNMcr,rcR->LNMR", scr3, ropr)
    return sgv0
    



def compute_sigmavector(mpo0, lopr, ropr, wfn0):
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

    scr1 = einsum('laL, lnr -> rnaL', lopr, wfn0)
    scr2 = einsum('rnaL, anNb -> rbNL', scr1, mpo0)
    sgv0 = einsum('rbNL, rbR -> LNR', scr2, ropr)
    return sgv0
	
def svd(idx, a, DMAX=0):
    """
    Thin Singular Value Decomposition

    idx : subscripts to split 
    a : ndarray
        matrix to do svd.
    DMAX: int
        maximal dim to keep.
     
    Returns
    -------
    u : ndarray
        left matrix
    s : ndarray
        sigular value
    vt : ndarray
        right matrix
    """
    idx0 = re.split(",", idx)
    assert len(idx0) == 2
    idx0[0].replace(" ", "")

    nsplit = len(idx0) 

    a = reshape(a, [np.prod(a.shape[:nsplit], -1]])
    u, s, vt = scipy.linalg.svd(a, full_matrices = False)
    
    M = len(s)
    if DMAX > 0:
        M = min(DMAX, M)

    u = u[:,:M]
    s = s[:M]
    vt = vt[:M,:]

    u = reshape(u, [a.shape[:nsplit] + [-1]])
    vt = reshape(vt, [a.shape[nsplit:]+[-1]])

    return u, s, vt
    

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
        mps0, s, wfn1 = svd("ij,k", wfn0, M)
        gaug = einsum("ij, jk -> ik", diag(s), wfn1)
    else:
        wfn1, s, mps0 = svd("i,jk", wfn0, M)
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
        upper MPS.
    ket0 : ndarray
        down MPS
     
    Returns
    -------
    opr1 : ndarray
        renormalized block opr.

    """
    
    if forward:
        scr1 = einsum('laL, LNR -> laNR', opr0, bra0.conj())
        scr2 = einsum('laNR, anNb-> lnbR ', scr1, mpo0)
        opr1 = einsum('lnbR, lnr-> rbR ', scr2, ket0)
    else:
        scr1 = einsum('LNR, rbR -> rbNL', bra0, opr0)
        scr2 = einsum('rbNL, anNb -> rnaL', scr1, mpo0)
        opr1 = einsum('rnaL, lnr-> laL ', scr2, ket0)
    
    return opr1    


def optimize_onesite(forward, mpo0, lopr, ropr, wfn0, wfn1, M = 0):
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
        MPS for canoicalization.
    wfn1 : ndarray
        MPS.
    M : int
        bond dimension
     
    Returns
    -------
    energy : float or list of floats
        The energy of desired root(s).

    """
    # def davidson(x0, diag_flat):
    #     """
    #     Davidson algorithm.

    #     Parameters
    #     ----------
    #     x0 : ndarray
    #         initial state.
    #     diag_flat : ndarray
    #         precomputed diagonal elements, 1D array.
         
    #     Returns
    #     -------
    #     energy : float or list of floats
    #         The energy of desired root(s).
    #     coeff : ndarray or list of ndarray
    #         The wavefunction.

    #     """

    diag_flat = compute_diagonal_elements(mpo0, lopr, ropr).ravel()
    
    mps_shape = wfn0.shape
    def compute_sigma_flat(x):
        return compute_sigmavector(mpo0, lopr, ropr, x.reshape(mps_shape)).ravel()
    def compute_precond_flat(dx, e, x0):
        return dx / (diag_flat - e)
    energy, wfn0 = lib.linalg_helper.davidson(compute_sigma_flat, wfn0.ravel(), compute_precond_flat)
    wfn0 = wfn.reshape(mps_shape)
    
    if forward:
        wfn0, gaug = canonicalize(1, wfn0, M) # wfn0 R => lmps gaug
        wfn1 = einsum("ij,jkl->ikl", gaug, wfn1)
        lopr = renormalize(1, mpo0, lopr, wfn0, wfn0)
    else:
        wfn0, gaug = canonicalize(0, wfn0, M) # wfn0 R => lmps gaug
        wfn1 = einsum("ijk,kl->ijl", wfn1, gaug)
        ropr = renormalize(0, mpo0, ropr, wfn0, wfn0)

    return energy, wfn0, wfn1, lopr, ropr

def sweep(mpos, mpss, lopr, ropr, algo = 'ONESITE', M = 1, tol = 1e-5):
    emin = 1.0e8
    print "\t++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    print "\t\t\tFORWARD SWEEP"
    print "\t++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    
    N = len(mpss)
    assert(len(mpos) == N);

    for i in xrange(N - 1):

        print "\t===================================================================================================="
        print "\t\tSITE [  %5d  ] "%i
        print "\t----------------------------------------------------------------------------------------------------"

        #print "\t\tloading operators and wavefunction of next site (env)..."
        #load(mpos[i+1], get_mpofile(input.prefix, i+1));
        #load(mpss[i+1], get_mpsfile(input.prefix, RIGHTCANONICAL, i+1));
        #cout << "done" << endl;
        sys.flush()

        # diagonalize
        if(algo == 'ONESITE'):
          print "\t\toptimizing wavefunction: 1-site algorithm "
          #load(ropr, get_oprfile(input.prefix, RIGHTCANONICAL, i))
          eswp = optimize_onesite(1, mpos[i], lopr, ropr, mpss[i], mpss[i + 1], M, 0.1 * tol)
          #eswp = optimize_onesite_merged(1, mpos[i], lopr, ropr, mpss[i], mpss[i+1], 0.1*tol, M) # ZHC NOTE

        else:
          print "\t\toptimizing wavefunction: 2-site algorithm "
          #load(ropr, get_oprfile(input.prefix, RIGHTCANONICAL, i+1));
          eswp = optimize_twosite(1, mpos[i], mpos[i + 1], lopr, ropr, mpss[i], mpss[i + 1], M, 0.1 * tol)
          #eswp = optimize_twosite_merged(1, mpos[i], mpos[i+1], lopr, ropr, mpss[i], mpss[i+1], 0.1*T, M);
        
        if(eswp < emin):
            emin = eswp

        # print result
        print "\t\t--------------------------------------------------------------------------------"
        print "\t\t\tEnergy = %20.10f "%eswp
        print "\t\t--------------------------------------------------------------------------------"

        #print "\t\tsaving operators and wavefunction of this site (sys)..."
        sys.flush()
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

    #for(size_t i = N-1; i > 0; --i) {
    #for i in xrange(N-1):
    for i in reversed(xrange(1, N)):
          
        print "\t===================================================================================================="
        print "\t\tSITE [  %5d  ] "%i
        print "\t----------------------------------------------------------------------------------------------------"

        #print "\t\tloading operators and wavefunction of next site (env)..."
        #load(mpos[i-1], get_mpofile(input.prefix, i-1));
        #load(mpss[i-1], get_mpsfile(input.prefix, LEFTCANONICAL, i-1));
        #print "done"

        # diagonalize
        if(algo == 'ONESITE'):
          print "\t\toptimizing wavefunction: 1-site algorithm "
          #load(ropr, get_oprfile(input.prefix, RIGHTCANONICAL, i))
          eswp = optimize_onesite(0, mpos[i], lopr, ropr, mpss[i], mpss[i - 1], M, 0.1 * tol)
          #eswp = optimize_onesite_merged(0, mpos[i],            lopr, ropr, mpss[i], mpss[i-1], 0.1*T, M);

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
    


def optimize_twosite(forward, lmpo, rmpo, lopr, ropr, lwfn, rwfn, M=0)
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
    diag = compute_diagonal_elements(lmpo, rmpo, lopr, ropr)

    mps_shape = wfn2.shape
    
    def compute_sigma_flat(x):
        return compute_sigmavector(mpo0, lopr, ropr, x.reshape(mps_shape)).ravel()
    def compute_precond_flat(dx, e, x0):
        return dx / (diag_flat - e)

    energy, wfn0 = lib.linalg_helper.davidson(compute_sigma_flat, wfn2.ravel(), compute_precond_flat)
    wfn0 = wfn0.reshape(mps_shape)

    if forward:
        wfn0, gaug = canonicalize(1, wfn0, M) # wfn0 R => lmps gaug
        wfn1 = einsum("ij,jkl->ikl", gaug, wfn1)
        lopr = renormalize(1, mpo0, lopr, wfn0, wfn0)
    else:
        wfn0, gaug = canonicalize(0, wfn0, M) # wfn0 R => lmps gaug
        wfn1 = einsum("ijk,kl->ijl", wfn1, gaug)
        ropr = renormalize(0, mpo0, ropr, wfn0, wfn0)

    return energy, wfn0, wfn1, lopr, ropr


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
    W.append(np.einsum('abnN -> anNb', np.array([[-h * Sz, 0.5 * J * Sm, 0.5 * J * Sp, J * Sz, I]])))
    for i in xrange(N-2):
        W.append(np.einsum('abnN -> anNb', np.array([[I, z, z, z, z],
                                                    [Sp, z, z, z, z],
                                                    [Sm, z, z, z, z],
                                                    [Sz, z, z, z, z],
                                                    [-h * Sz, 0.5 * J * Sm, 0.5 * J * Sp, J * Sz, I]])))
    W.append(np.einsum('abnN -> anNb', np.array([[I], [Sp], [Sm], [Sz], [-h * Sz]])))
    return W
