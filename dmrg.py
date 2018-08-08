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
	



def mps_dot(bra, ket):
    """
    Dot function for two mps.

    """
    return einsum('lnr, lnr', bra.conj(), ket)


def svd(arrow, a, DMAX=0): # NOTE move to quantum number tensor class 
    """
    Thin Singular Value Decomposition
    
    Parameters
    ----------
    arrow : enum obj
        Arrow.LeftArrow for left, Arrow.RightArrow for right.
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

    if arrow == Arrow.LeftArrow: #ZHC NOTE the arrow is a number? then should be .value
        a = reshape(a, [a.shape[0] * a.shape[1], -1])
        u, s, vt = scipy.linalg.svd(a, full_matrices = False)
    elif arrow == Arrow.RightArrow:
        a = reshape(a, [a.shape[0], -1])
        u, s, vt = scipy.linalg.svd(a, full_matrices = False)

    M = len(s)
    if DMAX > 0:
        M = min(DMAX, M)

    u = u[:,:M]
    s = s[:M]
    vt = vt[:M,:]

    if arrow == Arrow.LeftArrow:
        u = reshape(u, [a.shape[0], a.shape[1], -1])
    elif arrow == Arrow.RightArrow:
        vt = reshape(vt, [M, a.shape[1], -1])

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
        mps0, s, wfn1 = svd(Arrow.LeftArrow, wfn0, M)
        gaug = einsum("ij, jk -> ik", diag(s), wfn1)
    else:
        wfn1, s, mps0 = svd(Arrow.RightArrow, wfn0, M)
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
    def davidson(x0, diag_flat):
        """
        Davidson algorithm.

        Parameters
        ----------
        x0 : ndarray
            initial state.
        diag_flat : ndarray
            precomputed diagonal elements, 1D array.
         
        Returns
        -------
        energy : float or list of floats
            The energy of desired root(s).
        coeff : ndarray or list of ndarray
            The wavefunction.

        """
        mps_shape = x0.shape
        def compute_sigma_flat(x):
            return compute_sigmavector(mpo0, lopr, ropr, x.reshape(mps_shape)).ravel()
        def compute_precond_flat(dx, e, x0):
            return dx / (diag_flat - e)

        e, c = lib.linalg_helper.davidson(compute_sigma_flat, x0.ravel(), compute_precond_flat)
        
        return e, c.reshape(mps_shape)


    #diag = compute_diagonal_elements(mpo0, lopr, ropr)
    diag_flat = compute_diagonal_elements(mpo0, lopr, ropr).ravel()
    
    energy, wfn0 = davidson(wfn0, diag_flat)

    if forward:
        wfn0, gaug = canonicalize(1, wfn0, M) # wfn0 R => lmps gaug
        wfn1 = einsum("ij,jkl->ikl", gaug, wfn1)
        lopr = renormalize(1, mpo0, lopr, wfn0, wfn0)
    else:
        wfn0, gaug = canonicalize(0, wfn0, M) # wfn0 R => lmps gaug
        wfn1 = einsum("ijk,kl->ijl", wfn1, gaug)
        ropr = renormalize(0, mpo0, ropr, wfn0, wfn0)

    return energy, wfn0, wfn1, lopr, ropr


def sweep():
    pass


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
    
    energy, coeff = davidson(compute_sigmavector, diag, wfn0)
    
def initialize(mpos, mpss, lopr):
    pass

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
    



template<class Q>
double optimize_twosite
(bool forward, const btas::QSDArray<4, Q>& lmpo,
               const btas::QSDArray<4, Q>& rmpo,
                     btas::QSDArray<3, Q>& lopr,
                     btas::QSDArray<3, Q>& ropr,
                     btas::QSDArray<3, Q>& lwfn,
                     btas::QSDArray<3, Q>& rwfn,
               const double& T, const int& M = 0)
{
  using std::cout;
  using std::endl;
  using std::flush;
  using std::setw;
  using std::setprecision;
  using std::fixed;
  using std::scientific;

  time_stamp ts;

  cout << "\t\tcomputing 2-site wavefunction..." << flush;
  btas::QSDArray<4, Q> wfn2;
  btas::QSDgemm(btas::NoTrans, btas::NoTrans, 1.0, lwfn, rwfn, 1.0, wfn2);
  cout << "done ( " << fixed << setprecision(2) << setw(8) << ts.lap() << " sec. ) " << endl;

  cout << "\t\tcomputing diagonal elements..." << flush;
  btas::QSDArray<4, Q> diag(wfn2.q(), wfn2.qshape());
  compute_diagonal_elements(lmpo, rmpo, lopr, ropr, diag);
  cout << "done ( " << fixed << setprecision(2) << setw(8) << ts.lap() << " sec. ) " << endl;

  cout << "\t\toptimizing wavefunction (Davidson solver)..." << endl;
  davidson::Functor<4, Q> f_sigmavector;
  f_sigmavector = boost::bind(compute_sigmavector<Q>, lmpo, rmpo, lopr, ropr, _1, _2);
  cout << "\t\tdone ( " << fixed << setprecision(2) << setw(8) << ts.lap() << " sec. ) " << endl;

  double energy = davidson::diagonalize(f_sigmavector, diag, wfn2);

  if(forward) {
    cout << "\t\tdoing singular value decomposition on wavefunction..." << flush;
    canonicalize(1, wfn2, lwfn, rwfn, M);
    cout << "done ( " << fixed << setprecision(2) << setw(8) << ts.lap() << " sec. ) " << endl;

    cout << "\t\trenormalizing operators to the next..." << flush;
    btas::QSDArray<3, Q> lopr_tmp;
    renormalize(1, lmpo, lopr, lwfn, lwfn, lopr_tmp);
    lopr = lopr_tmp;
    cout << "done ( " << fixed << setprecision(2) << setw(8) << ts.lap() << " sec. ) " << endl;
  }
  else {
    cout << "\t\tdoing singular value decomposition on wavefunction..." << flush;
    canonicalize(0, wfn2, rwfn, lwfn, M);
    cout << "done ( " << fixed << setprecision(2) << setw(8) << ts.lap() << " sec. ) " << endl;

    cout << "\t\trenormalizing operators to the next..." << flush;
    btas::QSDArray<3, Q> ropr_tmp;
    renormalize(0, rmpo, ropr, rwfn, rwfn, ropr_tmp);
    ropr = ropr_tmp;
    cout << "done ( " << fixed << setprecision(2) << setw(8) << ts.lap() << " sec. ) " << endl;
  }
  cout << "\t\t--------------------------------------------------------------------------------" << endl;
  cout << "\t\tTotal time for optimization: " << fixed << setprecision(2) << setw(8) << ts.elapsed() << " sec. " << endl;

  return energy;
}

//
// Merged block version
//

template<class Q>
void compute_merged_block
(bool forward, const btas::QSDArray<4, Q>& mpo0,
                     btas::QSDArray<3, Q>& opr0,
                     btas::QSDArray<3, Q>& oprm)
{
  using std::cout;
  using std::endl;
  using std::flush;
  using std::setw;
  using std::setprecision;
  using std::fixed;
  using std::scientific;

  using btas::shape;

  if(forward) {
    const btas::Qshapes<Q>& q_l_bra = opr0.qshape(0);
    const btas::Qshapes<Q>& q_n_bra = mpo0.qshape(1);
    const btas::Dshapes&    d_l_bra = opr0.dshape(0);
    const btas::Dshapes&    d_n_bra = mpo0.dshape(1);

    const btas::Qshapes<Q>& q_l_ket = opr0.qshape(2);
    const btas::Qshapes<Q>& q_n_ket = mpo0.qshape(2);
    const btas::Dshapes&    d_l_ket = opr0.dshape(0);
    const btas::Dshapes&    d_n_ket = mpo0.dshape(1);

    btas::QSDArray<5, Q> lopr;
    btas::Contract(1.0, opr0, shape(0,1,2), mpo0, shape(1,3,4,5), 1.0, lopr, shape(0,3,5,2,4));

    btas::QSTmergeInfo<2, Q> q_mg_bra(btas::make_array(q_l_bra, q_n_bra), btas::make_array(d_l_bra, d_n_bra));
    btas::QSTmergeInfo<2, Q> q_mg_ket(btas::make_array(q_l_ket, q_n_ket), btas::make_array(d_l_ket, d_n_ket));

    btas::QSDArray<4, Q> oprx;
    btas::QSTmerge(q_mg_bra, lopr, oprx);
    btas::QSTmerge(oprx, q_mg_ket, oprm);
  }
  else {
    const btas::Qshapes<Q>& q_n_bra = mpo0.qshape(1);
    const btas::Qshapes<Q>& q_r_bra = opr0.qshape(0);
    const btas::Dshapes&    d_n_bra = mpo0.dshape(1);
    const btas::Dshapes&    d_r_bra = opr0.dshape(0);

    const btas::Qshapes<Q>& q_n_ket = mpo0.qshape(2);
    const btas::Qshapes<Q>& q_r_ket = opr0.qshape(2);
    const btas::Dshapes&    d_n_ket = mpo0.dshape(1);
    const btas::Dshapes&    d_r_ket = opr0.dshape(0);

    btas::QSDArray<5, Q> ropr;
    btas::Contract(1.0, mpo0, shape(0,1,2,3), opr0, shape(4,3,5), 1.0, ropr, shape(1,4,0,2,5));

    btas::QSTmergeInfo<2, Q> q_mg_bra(btas::make_array(q_n_bra, q_r_bra), btas::make_array(d_n_bra, d_r_bra));
    btas::QSTmergeInfo<2, Q> q_mg_ket(btas::make_array(q_n_ket, q_r_ket), btas::make_array(d_n_ket, d_r_ket));

    btas::QSDArray<4, Q> oprx;
    btas::QSTmerge(q_mg_bra, ropr, oprx);
    btas::QSTmerge(oprx, q_mg_ket, oprm);
  }
}

template<class Q>
double optimize_onesite_merged
(bool forward, const btas::QSDArray<4, Q>& mpo0,
                     btas::QSDArray<3, Q>& lopr,
                     btas::QSDArray<3, Q>& ropr,
                     btas::QSDArray<3, Q>& wfn0,
                     btas::QSDArray<3, Q>& wfn1,
               const double& T, const int& M = 0)
{
  using std::cout;
  using std::endl;
  using std::flush;
  using std::setw;
  using std::setprecision;
  using std::fixed;
  using std::scientific;

  btas::QSDArray<3, Q> lopr_mg;
  btas::QSDArray<3, Q> ropr_mg;
  btas::QSDArray<2, Q> wfnc_mg;
  btas::QSTmergeInfo<2, Q> q_mg_ket;

  time_stamp ts;

  cout << "\t\tconstructing merged super blocks..." << flush;
  if(forward) {
    const btas::Qshapes<Q>& q_l_ket =-lopr.qshape(2);
    const btas::Qshapes<Q>& q_n_ket =-mpo0.qshape(2);
    const btas::Dshapes&    d_l_ket = lopr.dshape(0);
    const btas::Dshapes&    d_n_ket = mpo0.dshape(1);
    q_mg_ket.reset(btas::make_array(q_l_ket, q_n_ket), btas::make_array(d_l_ket, d_n_ket));

    btas::QSTmerge(q_mg_ket, wfn0, wfnc_mg);

    compute_merged_block(1, mpo0, lopr, lopr_mg);
    ropr_mg.reference(ropr);
  }
  else {
    const btas::Qshapes<Q>& q_n_ket =-mpo0.qshape(2);
    const btas::Qshapes<Q>& q_r_ket =-ropr.qshape(2);
    const btas::Dshapes&    d_n_ket = mpo0.dshape(1);
    const btas::Dshapes&    d_r_ket = ropr.dshape(0);
    q_mg_ket.reset(btas::make_array(q_n_ket, q_r_ket), btas::make_array(d_n_ket, d_r_ket));

    btas::QSTmerge(wfn0, q_mg_ket, wfnc_mg);

    lopr_mg.reference(lopr);
    compute_merged_block(0, mpo0, ropr, ropr_mg);
  }
  cout << "done ( " << fixed << setprecision(2) << setw(8) << ts.lap() << " sec. ) " << endl;

  cout << "\t\tcomputing diagonal elements..." << flush;
  btas::QSDArray<2, Q> diag(wfnc_mg.q(), wfnc_mg.qshape());
  compute_diagonal_elements(lopr_mg, ropr_mg, diag);
  cout << "done ( " << fixed << setprecision(2) << setw(8) << ts.lap() << " sec. ) " << endl;

  cout << "\t\toptimizing wavefunction (Davidson solver)..." << endl;
  davidson::Functor<2, Q> f_sigmavector = boost::bind(compute_sigmavector<Q>, lopr_mg, ropr_mg, _1, _2);
  double energy = davidson::diagonalize(f_sigmavector, diag, wfnc_mg);
  cout << "\t\tdone ( " << fixed << setprecision(2) << setw(8) << ts.lap() << " sec. ) " << endl;

  if(forward) {
    cout << "\t\tdoing singular value decomposition on wavefunction..." << flush;
    btas::QSDArray<2, Q> lmps_mg;
    btas::QSDArray<2, Q> gaug;
    canonicalize(1, wfnc_mg, lmps_mg, gaug, M);
    cout << "done ( " << fixed << setprecision(2) << setw(8) << ts.lap() << " sec. ) " << endl;

    cout << "\t\texpanding wavefunction..." << flush;
    btas::QSTexpand(q_mg_ket, lmps_mg, wfn0);
    cout << "done ( " << fixed << setprecision(2) << setw(8) << ts.lap() << " sec. ) " << endl;

    cout << "\t\tcomputing guess wavefunction to the next..." << flush;
    btas::QSDArray<3, Q> wfn1_tmp;
    btas::QSDgemm(btas::NoTrans, btas::NoTrans, 1.0, gaug, wfn1, 1.0, wfn1_tmp);
    wfn1 = wfn1_tmp;
    cout << "done ( " << fixed << setprecision(2) << setw(8) << ts.lap() << " sec. ) " << endl;

    cout << "\t\trenormalizing operators to the next..." << flush;
    lopr.clear();
    renormalize(1, lopr_mg, lmps_mg, lmps_mg, lopr);
    cout << "done ( " << fixed << setprecision(2) << setw(8) << ts.lap() << " sec. ) " << endl;
  }
  else {
    cout << "\t\tdoing singular value decomposition on wavefunction..." << flush;
    btas::QSDArray<2, Q> rmps_mg;
    btas::QSDArray<2, Q> gaug;
    canonicalize(0, wfnc_mg, rmps_mg, gaug, M);
    cout << "done ( " << fixed << setprecision(2) << setw(8) << ts.lap() << " sec. ) " << endl;

    cout << "\t\texpanding wavefunction..." << flush;
    btas::QSTexpand(rmps_mg, q_mg_ket, wfn0);
    cout << "done ( " << fixed << setprecision(2) << setw(8) << ts.lap() << " sec. ) " << endl;

    cout << "\t\tcomputing guess wavefunction to the next..." << flush;
    btas::QSDArray<3, Q> wfn1_tmp;
    btas::QSDgemm(btas::NoTrans, btas::NoTrans, 1.0, wfn1, gaug, 1.0, wfn1_tmp);
    wfn1 = wfn1_tmp;
    cout << "done ( " << fixed << setprecision(2) << setw(8) << ts.lap() << " sec. ) " << endl;

    cout << "\t\trenormalizing operators to the next..." << flush;
    ropr.clear();
    renormalize(0, ropr_mg, rmps_mg, rmps_mg, ropr);
    cout << "done ( " << fixed << setprecision(2) << setw(8) << ts.lap() << " sec. ) " << endl;
  }
  cout << "\t\t--------------------------------------------------------------------------------" << endl;
  cout << "\t\tTotal time for optimization: " << fixed << setprecision(2) << setw(8) << ts.elapsed() << " sec. " << endl;

  return energy;
}

template<class Q>
double optimize_twosite_merged
(bool forward, const btas::QSDArray<4, Q>& lmpo,
               const btas::QSDArray<4, Q>& rmpo,
                     btas::QSDArray<3, Q>& lopr,
                     btas::QSDArray<3, Q>& ropr,
                     btas::QSDArray<3, Q>& lwfn,
                     btas::QSDArray<3, Q>& rwfn,
               const double& T, const int& M = 0)
{
  using std::cout;
  using std::endl;
  using std::flush;
  using std::setw;
  using std::setprecision;
  using std::fixed;
  using std::scientific;

  time_stamp ts;

  cout << "\t\tconstructing merged super blocks..." << flush;

  const btas::Qshapes<Q>& q_l_ket =-lopr.qshape(2);
  const btas::Qshapes<Q>& q_m_ket =-lmpo.qshape(2);
  const btas::Dshapes&    d_l_ket = lopr.dshape(0);
  const btas::Dshapes&    d_m_ket = lmpo.dshape(1);
  btas::QSTmergeInfo<2, Q> q_lmg_ket(btas::make_array(q_l_ket, q_m_ket), btas::make_array(d_l_ket, d_m_ket));

  btas::QSDArray<3, Q> lopr_mg;
  compute_merged_block(1, lmpo, lopr, lopr_mg);

  const btas::Qshapes<Q>& q_n_ket =-rmpo.qshape(2);
  const btas::Qshapes<Q>& q_r_ket =-ropr.qshape(2);
  const btas::Dshapes&    d_n_ket = rmpo.dshape(1);
  const btas::Dshapes&    d_r_ket = ropr.dshape(0);
  btas::QSTmergeInfo<2, Q> q_rmg_ket(btas::make_array(q_n_ket, q_r_ket), btas::make_array(d_n_ket, d_r_ket));

  btas::QSDArray<3, Q> ropr_mg;
  compute_merged_block(0, rmpo, ropr, ropr_mg);

  btas::QSDArray<2, Q> wfnc_mg;
  {
    btas::QSDArray<4, Q> wfn2;
    btas::QSDgemm(btas::NoTrans, btas::NoTrans, 1.0, lwfn, rwfn, 1.0, wfn2);
    btas::QSTmerge(q_lmg_ket, wfn2, q_rmg_ket, wfnc_mg);
  }

  cout << "done ( " << fixed << setprecision(2) << setw(8) << ts.lap() << " sec. ) " << endl;

  cout << "\t\tcomputing diagonal elements..." << flush;
  btas::QSDArray<2, Q> diag(wfnc_mg.q(), wfnc_mg.qshape());
  compute_diagonal_elements(lopr_mg, ropr_mg, diag);
  cout << "done ( " << fixed << setprecision(2) << setw(8) << ts.lap() << " sec. ) " << endl;

  cout << "\t\toptimizing wavefunction (Davidson solver)..." << endl;
  davidson::Functor<2, Q> f_sigmavector = boost::bind(compute_sigmavector<Q>, lopr_mg, ropr_mg, _1, _2);
  double energy = davidson::diagonalize(f_sigmavector, diag, wfnc_mg);
  cout << "\t\tdone ( " << fixed << setprecision(2) << setw(8) << ts.lap() << " sec. ) " << endl;

  btas::QSDArray<2, Q> lwfn_mg;
  btas::QSDArray<2, Q> rwfn_mg;

  if(forward) {
    cout << "\t\tdoing singular value decomposition on wavefunction..." << flush;
    canonicalize(1, wfnc_mg, lwfn_mg, rwfn_mg, M);
    cout << "done ( " << fixed << setprecision(2) << setw(8) << ts.lap() << " sec. ) " << endl;

    cout << "\t\texpanding wavefunction..." << flush;
    btas::QSTexpand(q_lmg_ket, lwfn_mg, lwfn);
    btas::QSTexpand(rwfn_mg, q_rmg_ket, rwfn);
    cout << "done ( " << fixed << setprecision(2) << setw(8) << ts.lap() << " sec. ) " << endl;

    cout << "\t\trenormalizing operators to the next..." << flush;
    lopr.clear();
    renormalize(1, lopr_mg, lwfn_mg, lwfn_mg, lopr);
    cout << "done ( " << fixed << setprecision(2) << setw(8) << ts.lap() << " sec. ) " << endl;
  }
  else {
    cout << "\t\tdoing singular value decomposition on wavefunction..." << flush;
    canonicalize(0, wfnc_mg, rwfn_mg, lwfn_mg, M);
    cout << "done ( " << fixed << setprecision(2) << setw(8) << ts.lap() << " sec. ) " << endl;

    cout << "\t\texpanding wavefunction..." << flush;
    btas::QSTexpand(q_lmg_ket, lwfn_mg, lwfn);
    btas::QSTexpand(rwfn_mg, q_rmg_ket, rwfn);
    cout << "done ( " << fixed << setprecision(2) << setw(8) << ts.lap() << " sec. ) " << endl;

    cout << "\t\trenormalizing operators to the next..." << flush;
    ropr.clear();
    renormalize(0, ropr_mg, rwfn_mg, rwfn_mg, ropr);
    cout << "done ( " << fixed << setprecision(2) << setw(8) << ts.lap() << " sec. ) " << endl;
  }
  cout << "\t\t--------------------------------------------------------------------------------" << endl;
  cout << "\t\tTotal time for optimization: " << fixed << setprecision(2) << setw(8) << ts.elapsed() << " sec. " << endl;

  return energy;
}

template<class Q>
double dmrg_sweep(MPOs<double, Q>& mpos, MPSs<double, Q>& mpss, const DmrgInput& input)
{
  using std::cout;
  using std::endl;
  using std::flush;
  using std::setw;
  using std::setprecision;
  using std::fixed;
  using std::scientific;

  size_t N = input.N_sites;
  int    M = input.N_max_states;
  double T = input.tolerance;

  assert(mpos.size() == N);
  assert(mpss.size() == N);

  btas::QSDArray<3, Q> lopr;
  btas::QSDArray<3, Q> ropr;

  double emin = 1.0e8;

  cout << "\t++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << endl;
  cout << "\t\t\tFORWARD SWEEP" << endl;
  cout << "\t++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << endl;

  load(mpos[0], get_mpofile(input.prefix, 0));
  load(mpss[0], get_mpsfile(input.prefix, WAVEFUNCTION,  0));
  load(lopr,    get_oprfile(input.prefix, LEFTCANONICAL, 0));

  for(size_t i = 0; i < N-1; ++i) {

    cout << "\t====================================================================================================" << endl;
    cout << "\t\tSITE [ " << setw(3) << i << " ] " << endl;
    cout << "\t----------------------------------------------------------------------------------------------------" << endl;

    cout << "\t\tloading operators and wavefunction of next site (env)..." << flush;
    load(mpos[i+1], get_mpofile(input.prefix, i+1));
    load(mpss[i+1], get_mpsfile(input.prefix, RIGHTCANONICAL, i+1));
    cout << "done" << endl;

    // diagonalize
    double eswp;
    if(input.algorithm == ONESITE) {
      cout << "\t\toptimizing wavefunction: 1-site algorithm " << endl;
      load(ropr, get_oprfile(input.prefix, RIGHTCANONICAL, i));
//    eswp = optimize_onesite(1, mpos[i],            lopr, ropr, mpss[i], mpss[i+1], 0.1*T, M);
      eswp = optimize_onesite_merged(1, mpos[i],            lopr, ropr, mpss[i], mpss[i+1], 0.1*T, M);
    }
    else {
      cout << "\t\toptimizing wavefunction: 2-site algorithm " << endl;
      load(ropr, get_oprfile(input.prefix, RIGHTCANONICAL, i+1));
//    eswp = optimize_twosite(1, mpos[i], mpos[i+1], lopr, ropr, mpss[i], mpss[i+1], 0.1*T, M);
      eswp = optimize_twosite_merged(1, mpos[i], mpos[i+1], lopr, ropr, mpss[i], mpss[i+1], 0.1*T, M);
    }
    if(eswp < emin) emin = eswp;
    // print result
    cout << "\t\t--------------------------------------------------------------------------------" << endl;
    cout.precision(16);
    cout << "\t\t\tEnergy = " << setw(24) << fixed << eswp << endl;
    cout << "\t\t--------------------------------------------------------------------------------" << endl;

    cout << "\t\tsaving operators and wavefunction of this site (sys)..." << flush;
    save(mpss[i], get_mpsfile(input.prefix, LEFTCANONICAL, i));
    save(lopr,    get_oprfile(input.prefix, LEFTCANONICAL, i+1));
    mpos[i].clear();
    mpss[i].clear();
    cout << "done" << endl;
  }
  save(mpss[N-1], get_mpsfile(input.prefix, WAVEFUNCTION, N-1));

  cout << "\t++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << endl;
  cout << "\t\t\tBACKWARD SWEEP" << endl;
  cout << "\t++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << endl;

  load(ropr, get_oprfile(input.prefix, RIGHTCANONICAL, N-1));

  for(size_t i = N-1; i > 0; --i) {

    cout << "\t====================================================================================================" << endl;
    cout << "\t\tSITE [ " << setw(3) << i << " ] " << endl;
    cout << "\t----------------------------------------------------------------------------------------------------" << endl;

    cout << "\t\tloading operators and wavefunction of next site (env)..." << flush;
    load(mpos[i-1], get_mpofile(input.prefix, i-1));
    load(mpss[i-1], get_mpsfile(input.prefix, LEFTCANONICAL, i-1));
    cout << "done" << endl;

    // diagonalize
    double eswp;
    if(input.algorithm == ONESITE) {
      cout << "\t\toptimizing wavefunction: 1-site algorithm " << endl;
      load(lopr,      get_oprfile(input.prefix, LEFTCANONICAL, i));
//    eswp = optimize_onesite(0, mpos[i],            lopr, ropr, mpss[i], mpss[i-1], 0.1*T, M);
      eswp = optimize_onesite_merged(0, mpos[i],            lopr, ropr, mpss[i], mpss[i-1], 0.1*T, M);
    }
    else {
      cout << "\t\toptimizing wavefunction: 2-site algorithm " << endl;
      load(lopr,      get_oprfile(input.prefix, LEFTCANONICAL, i-1));
//    eswp = optimize_twosite(0, mpos[i-1], mpos[i], lopr, ropr, mpss[i-1], mpss[i], 0.1*T, M);
      eswp = optimize_twosite_merged(0, mpos[i-1], mpos[i], lopr, ropr, mpss[i-1], mpss[i], 0.1*T, M);
    }
    if(eswp < emin) emin = eswp;
    // print result
    cout << "\t\t--------------------------------------------------------------------------------" << endl;
    cout.precision(16);
    cout << "\t\t\tEnergy = " << setw(24) << fixed << eswp << endl;
    cout << "\t\t--------------------------------------------------------------------------------" << endl;

    cout << "\t\tsaving operators and wavefunction for this site (sys)..." << flush;
    save(mpss[i], get_mpsfile(input.prefix, RIGHTCANONICAL, i));
    save(ropr,    get_oprfile(input.prefix, RIGHTCANONICAL, i-1));
    mpos[i].clear();
    mpss[i].clear();
    cout << "done" << endl;
  }
  save(mpss[0], get_mpsfile(input.prefix, WAVEFUNCTION, 0));
  mpos[0].clear();
  mpss[0].clear();

  cout << "\t====================================================================================================" << endl;

  return emin;
}

double dmrg(const DmrgInput& input);

}; // namespace mpsxx

#endif // _MPSXX_CXX11_DMRG_H
