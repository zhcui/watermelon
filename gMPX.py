import numpy as np
import generic_np
_einsum = generic_np.einsum
_dot = generic_np.dot
_diag = generic_np.diag
_svd = generic_np.svd
_sqrt = np.sqrt

"""
Generic MPX functions
"""

def loop_mpxs(mpxs, func = None, collect = True, *args):
    """
    Function to loop over MPXs, and print / apply function on each MPX.

    Parameters
    ----------

    Returns
    -------
    """
    if func is None:
        for mpx in mpxs:
            print mpx
        return
    if collect:
        return [func(mpx, *args) for mpx in mpxs]
    else:
        for mpx in mpxs:
            func(mpx, *args)

def create(dp, D=None, bc = None, fn=None, dtype=None, fix_D = False):
    # TODO: currently if D!=None and pbc, guarantees
    # all bond dims are D; but if obc, then
    # always follows obc_dim. Add option for guaranteed
    # bond dim D even for OBC
    """
    Create random MPX object as ndarray of ndarrays

    Parameters
    ----------
    dp : list of ints or list of 2-tuples
      Specifies physical dimensions of MPS or MPO. 
    D : int, maximum bond dimension

    Returns
    -------
    mpx : ndarray of ndarrays
       MPS or MPO
    """
    L = len(dp)
    mpx = np.empty(L, dtype=np.object)

    try: # if MPO: get flattened phys. dim.
        if len(dp[0]) == 2: 
            _dp = [p0 * p1 for p0, p1 in dp]
    except:
        _dp = dp
        

    # fill in MPX with arrays of the correct shape
    if bc == "obc":
        # calculate right bond dims of each tensor
        dim_rs = obc_dim(_dp, D, fix_D = fix_D)
        mpx[0]  = fn((1, _dp[0], dim_rs[0]), dtype=dtype)
        for i in range(1, L-1):
            mpx[i] = fn((dim_rs[i-1], _dp[i], dim_rs[i]), dtype=dtype)
        mpx[-1] = fn((dim_rs[-1], _dp[-1], 1), dtype=dtype)

    elif bc == "pbc":
        for i in range(L):
            mpx[i] = fn((D, _dp[i], D), dtype=dtype)
    else:
        raise RuntimeError, "bc not specified"
    
    try: # if MPO: reshape flattened phys. dim.
        if len(dp[0]) == 2: 
            for i in range(L):
                mpx[i] = np.reshape(mpx[i], [mpx[i].shape[0],
                                             dp[i][0], dp[i][1],
                                             mpx[i].shape[2]])
    except:
        pass
            
    return mpx

def obc_dim(dp, D = None, fix_D = False):
    """
    Right bond dimensions for OBC MPX

    Parameters
    ----------
    dp : sequence of int
      Physical dimension of MPX
    D  : int, max bond dimension

    Returns
    -------
    dimMin : list of int, right bond dimensions
    """
    if fix_D == True:
        if D is None:
            raise ValueError
        dimMin = np.asarray([D] * len(dp))
        return dimMin

    print 'gMPX, dp = ', dp
    print 'gMPX, len(dp) = ', len(dp)

    N = len(dp)
    dimR = D * np.ones(N)
    dimL = D * np.ones(N)
    
    dR = 1
    for i in xrange(N):
        dR *= dp[i]
        if dR < D:
            dimR[i] = dR
        else:
            break
    dL = 1
    for i in xrange(N):
        dL *= dp[N-i-1]
        if dL < D:
            dimL[N-i-1] = dL
        else:
            break
    print 'dimR = ', dimR
    print 'dimL = ', dimL
    dimMin = np.minimum(dimR[:-1], dimL[1:])
#   TODO: Take care of the case when D = None
#    if D is not None:
#        dimMin = np.minimum(dimMin, [D] * (len(dp) - 1))
    return dimMin.astype(int)

def element(mpx, occ, bc=None):
    """
    Evaluate MPX for specified physical indices

    Parameters
    ----------
    dp : sequence of int
      Physical dimension of MPX
    occ : sequence of int (MPS) / tuple[2] (MPO)
      Physical index

    Returns
    -------
    elements: 2D ndarray (pbc) / scalar (obc)
    """
    mats = [None] * len(mpx)
    try: # mpx is an mpo
        if len(occ[0]) == 2:
            for i, m in enumerate(mpx):
                mats[i] = mpx[i][:,occ[i][0],occ[i][1],:]
    except:
        for i, m in enumerate(mpx):
            mats[i] = mpx[i][:,occ[i],:]

    element = mats[0]
    for i in range(1, len(mpx)):
        element = _dot(element, mats[i])
    return _einsum("i...i", element)

def asfull(mpx):
    """
    Return full Hilbert space representation
    """
    dp = tuple([m.shape[1] for m in mpx])

    n = np.prod(dp)
    dtype = mpx[0].dtype
    if mpx[0].ndim == 4: # mpx is an mpo
        dense = np.zeros([n, n], dtype=dtype)
        for occi in np.ndindex(dp):
            i = np.ravel_multi_index(occi, dp)
            for occj in np.ndindex(dp):
                j = np.ravel_multi_index(occj, dp)
                dense[i, j] = element(mpx, zip(occi, occj))
    else:
        assert mpx[0].ndim == 3 # mpx is an mpo
        dense = np.zeros([n], dtype=dtype)        
        for occi in np.ndindex(dp):
            i = np.ravel_multi_index(occi, dp)
            dense[i] = element(mpx, occi)
            
    return dense

def mul(alpha, mpx):
    """
    Scale MPX by alpha

    Returns
    -------
    new_mpx : scaled MPX
    """
    L = len(mpx)
    new_mpx = np.empty(L,dtype=np.object)

    const = np.abs(alpha)**(1./L)
    dtype = np.result_type(alpha,mpx[0])
    for i in range(L):
        new_mpx[i] = mpx[i] * const
        #new_mpx[i] = np.array(mpx[i],dtype=dtype)

    # restore phase on first tensor
    new_mpx[0] *= (alpha/np.abs(alpha))
    #new_mpx[0] *= (alpha)

    return new_mpx
    
def compress(mpx0, D, preserve_dim=False, direction=0):
    """
    Compress MPX to dimension D

    Parameters
    ----------
    mpx0 : MPS or MPO
    D : int
      max dimension to compress to
    preserve_dim : bool
      if True, then the dimensions of the mpx tensors
      are unchanged; truncation sets elements to zero
    """
    tot_dwt = 0
    L = len(mpx0)

    mpx = mpx0.copy()
    preserve_uv = None
    if direction == 0:
        if preserve_dim:
            preserve_uv = "u"
        for i in range(L-1):
            u, s, vt, dwt = _svd("ij,k", mpx[i], D, preserve_uv)
            #tot_dwt += dwt
            mpx[i] = u
            svt = _dot(_diag(s), vt)
            mpx[i+1] = _einsum("lj,jnr->lnr", svt, mpx[i+1])
    else:
        if preserve_dim:
            preserve_uv = "v"
        for i in range(L-1,0,-1):
            u, s, vt, dwt = _svd("i,jk", mpx[i], D, preserve_uv)
            #tot_dwt += dwt
            mpx[i] = vt
            us = _dot(u, _diag(s))
            mpx[i-1] = _einsum("lnj,jr->lnr",  mpx[i-1], us)

    return mpx

def dot(mpx1, mpx2):
    """
    Computes MPX * MPX

    Parameters
    ----------
    mpx1: MPO or MPS
    mpx2: MPO or MPS

    Returns
    -------
     new_mpx : float or MPS or MPO
    """

    L = len(mpx1)
    assert len(mpx2)==L, '[dot]: lengths of mpx1 and mpx2 are not equal'
    new_mpx = np.empty(L, dtype=np.object)

    if mpx1[0].ndim == 3 and mpx2[0].ndim == 3:
        return _mps_dot(mpx1, mpx2)
    
    elif mpx1[0].ndim == 4 and mpx2[0].ndim == 3:
        for i in range(L):
            new_site = _einsum('LNnR,lnr->LlNRr',mpx1[i],mpx2[i])
            sh = new_site.shape
            new_mpx[i] = new_site.reshape((sh[0]*sh[1], sh[2], -1))

    elif mpx1[0].ndim == 3 and mpx2[0].ndim == 4:
        for i in range(L):
            new_site = _einsum('LNR,lNnr->LlnRr',mpx1[i],mpx2[i])
            sh = new_site.shape
            new_mpx[i] = new_site.reshape((sh[0]*sh[1], sh[2], -1))
            
    elif mpx1[0].ndim == 4 and mpx2[0].ndim == 4:
        for i in range(L):
            new_site = _einsum('LNMR,lMnr->LlNnRr',mpx1[i],mpx2[i])
            sh = new_site.shape
            new_mpx[i] = new_site.reshape((sh[0]*sh[1],sh[2],sh[3],-1))

    else:
        raise NotImplementedError('mpx of dim', mpx2[0].ndim, 'has not yet been implemented')

    return new_mpx

def norm(mpx): 
    """
    2nd norm of a MPX

    Parameters
    ----------
    mpx : MPS or MPO

    Returns
    -------
    norm : scalar
    """
    norm_val = vdot(flatten(mpx),flatten(mpx))
    # catch cases when norm is ~0 but in reality is a small negative number
    assert(norm_val > -1.0e-12), norm_val
    return _sqrt(np.abs(norm_val))

def flatten(mpx):
    """
    Converts MPX object into MPS

    Parameters
    ----------
    mpx : MPS or MPO

    Returns
    -------
    mps : MPS
    """
    if mpx[0].ndim == 3: # already MPS
        return mpx
    else: # MPO
        assert mpx[0].ndim == 4
        L = len(mpx)
        mps = []
        for i in range(L):
            sh = mpx[i].shape
            mps.append(np.reshape(mpx[i], (sh[0], sh[1]*sh[2], -1)))
        return np.asarray(mps)

def vdot(mps1, mps2, direction=0):
    """
    vdot of two MPS, returns scalar

    cf. np.vdot
    """
    return _mps_dot(mps1.conj(), mps2, direction)
    #return _mps_dot(mps1, mps2, direction)

def _mps_dot(mps1, mps2, direction=0, trace=True):
    """
    dot of two MPS, returns scalar
    """
    L = len(mps1)
    assert len(mps2) == L and direction in (0, 1)
    
    if direction == 0:
        mps1_ = mps1
        mps2_ = mps2
    elif direction == 1:       # contract right to left
        mps1_ = mps1[::-1]
        mps2_ = mps2[::-1]
        
    E = _einsum('InR, inr -> IiRr', mps1_[0], mps2_[0]) 
    #E = einsum('InR, inr -> IirR', mps1_[0], mps2_[0]) 
    for i in xrange(1, L):
        # contract with bra
        E = _einsum('IiRr, RnL -> IirnL', E, mps1_[i])
        #E = einsum('IirR, RnL -> IirnL', E, mps1_[i])
        # contract with ket
        E = _einsum('IirnL, rnl -> IiLl', E, mps2_[i])

    if trace:
        return _einsum('ijij', E)
    else:
        return E



############################################
# NOT TESTED
def unflatten(mpx):
    """
    Converts MPX object into MPO

    Parameters
    ----------
    mpx : MPS or MPO

    Returns
    -------
    mpo : MPO
    """
    if mpx[0].ndim == 4: # already MPO
        return mpx
    else:
        assert mpx[0].ndim == 3
        L = len(mpx)
        mpo = np.empty([L], dtype=np.object)
        for i in range(L):
            sh = mpx[i].shape
            p = int(sqrt(mpx[i].shape[1]))
            mpo[i] = np.reshape(mpx[i], (sh[0], p, p, -1))
        return mpo

def inprod(mps1, mpo, mps2, direction=0):
    """
    Computes <MPS1 | MPO | MPS2>
    
    Note: bra is not conjugated, and
          MPS1, MPS2 assumed to have OBC

    Parameters
    ----------
    mps1 : MPS
    mpo : MPO
    mps2 : MPS

    Returns
    -------
    inprod : float

    """
    assert direction in (0, 1)
    
    if direction == 0:     # contract left to right
       mps1_ = mps1
       mpo_  = mpo
       mps2_ = mps2
    elif direction == 1:       # contract right to left
       mps1_ = mps1[::-1]
       mpo_  = mpo[::-1]
       mps2_ = mps2[::-1]
   
    E = _einsum('lnr,anNb,LNR->rbR',
                  mps1[0], mpo[0], mps2[0])
    for i in range(1,L):
        E = _einsum('rbR,lnr,anNb,LNR',
                      E, mps1[i], mpo[i], mps2[i])

    return _einsum('i...i', E)
                    

def dot_compress(mpx1,mpx2,D,direction=0):
    # returns mpx1*mpx2 (ie mpsx1 applied to mpsx2) in mpx form, with compression of each bond

    L = len(mpx1)
    assert(len(mpx2)==L)
    new_mpx = np.empty(L,dtype=np.object)
    tot_dwt = 0

    if not direction == 0:
        mpx1 = [np.swapaxes(m,0,-1) for m in mpx1[::-1]]   # taking the left/right transpose
        mpx2 = [np.swapaxes(m,0,-1) for m in mpx2[::-1]]
    else:
        mpx1 = mpx1
        mpx2 = mpx2

    if mpx1[0].ndim == 3 and mpx2[0].ndim == 3:
        return _mps_dot(mpx1, mpx2)
    
    elif mpx1[0].ndim == 4 and mpx2[0].ndim == 3:
        prev_site = _einsum('LNnR,lnr->LlNRr',mpx1[0],mpx2[0])
        prev_site = linalg.reshape(prev_site,'ab,c,de')
        for i in range(1,L):
            new_site = _einsum('LNnR,lnr->LlNRr',mpx1[i],mpx2[i])
            new_site = linalg.reshape(new_site,'ab,c,de')
            temp_mpx = np.empty(2,dtype=np.object)
            temp_mpx[:] = [prev_site,new_site]
            [new_mpx[i-1],prev_site],dwt = compress(temp_mpx,D)
            tot_dwt += dwt
        new_mpx[-1] = prev_site

    elif mpx1[0].ndim == 3 and mpx2[0].ndim == 4:
        prev_site = _einsum('LNR,lNnr->LlnRr',mpx1[0],mpx2[0])
        prev_site = linalg.reshape(prev_site,'ab,c,de')
        for i in range(1,L):
            new_site = _einsum('LNR,lNnr->LlnRr',mpx1[i],mpx2[i])
            new_site = linalg.reshape(new_site,'ab,c,de')
            temp_mpx = np.empty(2,dtype=np.object)
            temp_mpx[:] = [prev_site,new_site]
            [new_mpx[i-1],prev_site],dwt = compress(temp_mpx,D)
            tot_dwt += dwt
        new_mpx[-1] = prev_site
            
    elif mpx1[0].ndim == 4 and mpx2[0].ndim == 4:
        prev_site = _einsum('LNMR,lMnr->LlNnRr',mpx1[0],mpx2[0])
        prev_site = linalg.reshape(prev_site,'ab,cd,ef')
        for i in range(1,L):
            new_site = _einsum('LNMR,lMnr->LlNnRr',mpx1[i],mpx2[i])
            new_site = linalg.reshape(new_site,'ab,cd,ef')
            temp_mpx = np.empty(2,dtype=np.object)
            temp_mpx[:] = [prev_site,new_site]
            shps = [m.shape for  m in temp_mpx]
            temp_mpx = flatten(temp_mpx)
            mpo_out,dwt = compress(temp_mpx,D)
            new_mpx[i-1] = mpo_out[0].reshape(shps[0][:-1]+(-1,))
            prev_site    = mpo_out[1].reshape((-1,)+shps[1][1:])
            tot_dwt += dwt
        new_mpx[-1] = prev_site

    else:
        raise NotImplementedError('mpx of dim', mpx2[0].ndim, 'has not yet been implemented')

    return new_mpx, tot_dwt
