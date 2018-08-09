import numpy as np

from numpy import einsum, dot, sqrt
import linalg

def create(dp, D=None, fn=np.zeros):
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
        
    # calculate right bond dims of each tensor
    dim_rs = calc_dim(_dp,D)

    # fill in MPX with random arrays of the correct shape
    mpx[0]  = fn((1, _dp[0], dim_rs[0]))
    for i in range(1, L-1):
        mpx[i] = fn((dim_rs[i-1], _dp[i], dim_rs[i]))
    mpx[-1] = fn((dim_rs[-1], _dp[-1], 1))

    try: # if MPO: reshape flattened phys. dim.
        if len(dp[0]) == 2: 
            for i in range(L):
                mpx[i] = np.reshape(mpx[i], [mpx[i].shape[0],
                                             dp[i][0], dp[i][1],
                                             mpx[i].shape[2]])
    except:
        pass
            
    return mpx

def empty(dp, D = None):
    return create(dp, D, fn=np.empty)

def zeros(dp, D = None):
    return create(dp, D, fn=np.zeros)

def rand(dp, D = None):
    return create(dp, D, fn=np.random.random)


def calc_dim(dps,D=None):
    # dps is a list/array of integers specifying the dimension of the physical bonds at each site
    # cap MPS to virtual dimension D
    # returns a list of the the right bond dimensions to be used in generating the MPS

    dimR = np.cumprod(dps)
    dimL = np.cumprod(dps[::-1])[::-1]
    
    dimMin = np.minimum(dimR[:-1],dimL[1:])
    if D is not None:
        dimMin = np.minimum(dimMin,[D]*(len(dps)-1))

    return dimMin

def element(mpx, occ):
    mats = [None] * len(mpx)
    try: # mpx is an mpo
        if len(occ[0]) == 2:
            for i, m in enumerate(mpx):
                mats[i] = mpx[i][:,occ[i][0],occ[i][1],:]
    except:
        for i, m in enumerate(mpx):
            mats[i] = mpx[i][:,occ[i],:]
        
    return np.asscalar(reduce(np.dot, mats))

def asfull(mpx):
    dp = tuple(m.shape[1] for m in mpx)

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


def product_state(dp, occ):
    # dp:  list/array of integers specifying dimension of physical bonds at each site
    # occ:  occupancy vector (len L), numbers specifying physical bond index occupied
    # returns product state mps according to occ
    L = len(dp)
    mps = zeros(dp, 1)
    for i in range(L):
        mps[i][0, occ[i], 0] = 1.

    return mps


def mul(alpha, mps):
    # result:  mps scaled by alpha

    L = len(mps)
    new_mps = np.empty(L,dtype=np.object)

    const = np.abs(alpha)**(1./L)
    dtype = np.result_type(alpha,mps[0])
    for i in range(L):
        new_mps[i] = np.array(mps[i],dtype=dtype)*const

    # change sign as specified by alpha
    try:     phase = np.sign(alpha)
    except:  phase = np.exp(1j*np.angle(alpha))

    new_mps[0] *= phase

    return new_mps


def add(mpx1, mpx2):
    L = len(mpx1)

    new_mpx = np.empty(L, dtype=np.object)

    dtype = np.result_type(mpx1[0], mpx2[0])

    assert len(mpx1)==len(mpx2), 'need to have same lengths: (%d,%d)'%(len(mpx1),len(mpx2))

    for i in range(L):
        sh1 = mpx1[i].shape
        sh2 = mpx2[i].shape
        assert sh1[1:-1]==sh2[1:-1], 'need physical bonds at site %d to match'%(i)        

        l1,n1,r1 = sh1[0],np.prod(sh1[1:-1]),sh1[-1]
        l2,n2,r2 = sh2[0],np.prod(sh2[1:-1]),sh2[-1]

        new_site = np.zeros((l1+l2,n1,r1+r2),dtype=dtype)
        new_site[:l1,:,:r1] = mpx1[i].reshape(l1,n1,r1)
        new_site[l1:,:,r1:] = mpx2[i].reshape(l2,n2,r2)

        new_site = new_site.reshape((l1+l2,)+sh1[1:-1]+(r1+r2,))

        if i==0:    new_site = np.einsum('l...r->...r',new_site).reshape((1,)+sh1[1:-1]+(r1+r2,))
        if i==L-1:  new_site = np.einsum('l...r->l...',new_site).reshape((l1+l2,)+sh1[1:-1]+(1,))

        new_mps[i] = new_site.copy()
    
    return new_mps

def axpby(alpha,mpx1,beta,mpx2):
    # GKC: reimplement in terms of add and scal
    # alpha = scalar, mps1,mps2 are ndarrays of tensors   
    # returns alpha*mps1 + mps2

    L = len(mpx1)
    mps_new = np.empty(L,dtype=np.object)
    
    const_a = np.abs(alpha)**(1./L)
    const_b = np.abs(beta)**(1./L)
    dtype = np.result_type(alpha,mpx1[0],beta,mpx2[0])
    assert(len(mpx1)==len(mpx2)), 'need to have same lengths: (%d,%d)'%(len(mpx1),len(mpx2))
    for i in range(len(mpx1)):
        sh1 = mpx1[i].shape
        sh2 = mpx2[i].shape
        assert(sh1[1:-1]==sh2[1:-1]), 'need physical bonds at site %d to match'%(i)        

        l1,n1,r1 = sh1[0],np.prod(sh1[1:-1]),sh1[-1]
        l2,n2,r2 = sh2[0],np.prod(sh2[1:-1]),sh2[-1]

        if i==0: 
            try:         sign_a, sign_b = np.sign(alpha,beta)
            except:      sign_a, sign_b = np.exp(1j*np.angle(alpha)), np.exp(1j*np.angle(beta))
        else:
            sign_a, sign_b = 1,1

        newSite = np.zeros((l1+l2,n1,r1+r2),dtype=dtype)
        newSite[:l1,:,:r1] = mpx1[i].reshape(l1,n1,r1)*const_a*sign_a
        newSite[l1:,:,r1:] = mpx2[i].reshape(l2,n2,r2)*const_b*sign_b

        newSite = newSite.reshape((l1+l2,)+sh1[1:-1]+(r1+r2,))

        if i==0:    newSite = np.einsum('l...r->...r',newSite).reshape((1,)+sh1[1:-1]+(r1+r2,))
        if i==L-1:  newSite = np.einsum('l...r->l...',newSite).reshape((l1+l2,)+sh1[1:-1]+(1,))

        mps_new[i] = newSite.copy()
    
    return mps_new
    
def compress(mpx0, D, direction=0):
    tot_dwt = 0
    L = len(mpx0)

    mpx = mpx0.copy()
    
    if direction == 0:
        for i in range(L-1):
            u, s, vt, dwt = linalg.svd("ij,k", mpx[i], D)
            tot_dwt += dwt
            mpx[i] = u
            svt = np.dot(np.diag(s), vt)
            mpx[i+1] = einsum("lj,jnr", svt, mpx[i+1])
    else:
        for i in range(L-1,0,-1):
            u, s, vt, dwt = linalg.svd("i,jk", mpx[i], D)
            tot_dwt += dwt
            mpx[i] = vt
            us = np.dot(u,np.diag(s))
            mpx[i-1] = einsum("lnj,jr",  mpx[i-1], us)

    return mpx, tot_dwt

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
   
    E = np.einsum('lnr,anNb,LNR->rbR',
                  mps1[0], mpo[0], mps2[0])
    for i in range(1,L):
        E = np.einsum('rbR,lnr,anNb,LNR',
                      E, mps1[i], mpo[i], mps2[i])

    return np.asscalar(E)

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
        return mps_dot(mpx1, mpx2)
    
    elif mpx1[0].ndim == 4 and mpx2[0].ndim == 3:
        for i in range(L):
            new_site = einsum('LNnR,lnr->LlNRr',mpx1[i],mpx2[i])
            sh = new_site.shape
            new_mpx[i] = new_site.reshape((sh[0]*sh[1], sh[2], -1))

    elif mpx1[0].ndim == 3 and mpx2[0].ndim == 4:
        for i in range(L):
            new_site = einsum('LNR,lNnr->LlnRr',mpx1[i],mpx2[i])
            sh = new_site.shape
            new_mpx[i] = new_site.reshape((sh[0]*sh[1], sh[2], -1))
            
    elif mpx1[0].ndim == 4 and mpx2[0].ndim == 4:
        for i in range(L):
            new_site = einsum('LNMR,lMnr->LlNnRr',mpx1[i],mpx2[i])
            sh = new_site.shape
            new_mpx[i] = new_site.reshape((sh[0]*sh[1],sh[2],sh[3],-1))

    else:
        raise NotImplementedError, 'mpx of dim', mpx2[0].ndim, 'has not yet been implemented'

    return new_mpx

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
        mps = np.empty_like(L)
        for i in L:
            sh = mpx[i].shape
            mps[i] = np.reshape(mpx[i], (sh[0], sh[1]*sh[2], -1))
        return mps

def dot_compress():
    pass

def vdot(mps1, mps2, direction=0):
    """
    vdot of two MPS, returns scalar

    cf. np.vdot
    """
    return mps_dot(np.conj(mps1), mps2, direction)

def mps_dot(mps1, mps2, direction=0):
    """
    dot of two MPS, returns scalar
    """
    L = len(mps1)
    assert len(mps2) == L
    assert direction in (0, 1)
    
    if direction == 0:
        E = einsum('lnR, lnr -> rR', mps1[0], mps2[0]) 
        for i in xrange(1, L):
            # contract with bra
            E = einsum('rR, RnL -> rnL', E, mps1[i])
            # contract with ket
            E = einsum('rnL, rnl -> lL', E, mps2[i])
    else:
        E = einsum('Lnr, lnr -> lL', mps1[-1], mps2[-1]) 
        for i in xrange(L - 1, -1, -1):
            # contract with bra
            E = einsum('lL, RnL -> lnR', E, mps1[i])
            # contract with ket
            E = einsum('lnR, rnl -> rR', E, mps2[i])
    return np.asscalar(E)

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
    return np.sqrt(vdot(flatten(mpx), flatten(mpx)))

