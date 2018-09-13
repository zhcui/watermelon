import sparse as sp

import pbc_MPS as MPS
MPS.dot = sparse.coo.dot


def zeros(dp, D = None, bc = None):
    def _from_np_zeros(shape):
        return sp.COO.from_numpy(np.zeros(shape))
    return MPS.create(dp, D, bc, fn=_from_np_zeros)

def rand(dp, D = None, bc = None, seed = None, density = 0.1):
    if seed is not None:
        np.random.seed(seed)
    return MPS.create(dp, D, bc, fn=sp.random)

def _ndarray(lst):
    L = len(lst)
    a = np.empty([L], np.object)
    for i in range(L):
        a[i] = lst[i]
    return a
    
def todense(mpx):
    """
    Convert all sparse tensors into dense tensors
    """
    return _ndarray([m.todense() for m in mpx])

def from_dense(mpx):
    """
    Convert from dense MPS representation
    """
    return _ndarray([sp.COO.from_numpy(m) for m in mpx])

def overwrite(mpx, out=None):
    """
    Overwrites tensors of mpx2 with tensors of mpx1,
    with fixed shape of mpx2 tensors.

    Parameters
    ----------
    mpx : MPX (source)
    out : MPX (target) [modified]
    """
    L = len(mpx)
    for i in range(L):
        m1 = sp.DOK.from_coo(out[i])
        m2 = sp.DOK(mpx[i])
        for coord in m2.data:
            m1[coord] = m2[coord]
        out[i] = m1.to_coo()
            

def add(mpx1, mpx2):
    """
    Add MPX1 to MPX2

    Parameters
    ----------
    mpx1, mpx2 : MPS or MPO

    Returns
    -------
    new_mpx : sum of MPS
    """
    L = len(mpx1)

    new_mpx = np.empty(L, dtype=np.object)
    dtype = np.result_type(mpx1[0], mpx2[0])

    assert len(mpx1)==len(mpx2), 'need to have same lengths: (%d,%d)' % (len(mpx1), len(mpx2))
    
    for i in range(L):
        sh1 = mpx1[i].shape
        sh2 = mpx2[i].shape
        assert sh1[1: -1] == sh2[1: -1], 'need physical bonds at site %d to match' % (i)        

        l1,n1,r1 = sh1[0],np.prod(sh1[1:-1]),sh1[-1]
        l2,n2,r2 = sh2[0],np.prod(sh2[1:-1]),sh2[-1]

        if i==0:
            new_site = np.zeros((max(l1,l2),n1,r1+r2),dtype=dtype)
            new_site[:l1,:,:r1] = mpx1[i].reshape(l1,n1,r1)
            new_site[:l2,:,r1:] = mpx2[i].reshape(l2,n2,r2)
        elif i==L-1:
            new_site = np.zeros((l1+l2,n1,max(r1,r2)),dtype=dtype)
            new_site[:l1,:,:r1] = mpx1[i].reshape(l1,n1,r1)
            new_site[l1:,:,:r2] = mpx2[i].reshape(l2,n2,r2)
        else:
            new_site = np.zeros((l1+l2,n1,r1+r2),dtype=dtype)
            new_site[:l1,:,:r1] = mpx1[i].reshape(l1,n1,r1)
            new_site[l1:,:,r1:] = mpx2[i].reshape(l2,n2,r2)

        nsh = new_site.shape
        new_site = new_site.reshape((nsh[0],)+sh1[1:-1]+(nsh[-1],))
        new_mpx[i] = new_site
    
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
            u, s, vt, dwt = linalg.svd("ij,k", mpx[i], D, preserve_uv)
            tot_dwt += dwt
            mpx[i] = u
            svt = np.dot(np.diag(s), vt)
            mpx[i+1] = einsum("lj,jnr", svt, mpx[i+1])
    else:
        if preserve_dim:
            preserve_uv = "v"
        for i in range(L-1,0,-1):
            u, s, vt, dwt = linalg.svd("i,jk", mpx[i], D, preserve_uv)
            tot_dwt += dwt
            mpx[i] = vt
            us = np.dot(u,np.diag(s))
            mpx[i-1] = einsum("lnj,jr",  mpx[i-1], us)

    return mpx, tot_dwt


####################################

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

    return np.einsum('i...i', E)

#####################
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
        raise NotImplementedError('mpx of dim', mpx2[0].ndim, 'has not yet been implemented')

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
        mps = []
        for i in range(L):
            sh = mpx[i].shape
            mps.append(np.reshape(mpx[i], (sh[0], sh[1]*sh[2], -1)))
        return np.asarray(mps)

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
        mpo = []
        for i in range(L):
            sh = mpx[i].shape
            p = int(sqrt(mpx[i].shape[1]))
            mpo.append(np.reshape(mpx[i], (sh[0], p, p, -1)))
        return np.asarray(mpo)
                    
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
        prev_site = einsum('LNnR,lnr->LlNRr',mpx1[0],mpx2[0])
        prev_site = linalg.reshape(prev_site,'ab,c,de')
        for i in range(1,L):
            new_site = einsum('LNnR,lnr->LlNRr',mpx1[i],mpx2[i])
            new_site = linalg.reshape(new_site,'ab,c,de')
            temp_mpx = np.empty(2,dtype=np.object)
            temp_mpx[:] = [prev_site,new_site]
            [new_mpx[i-1],prev_site],dwt = compress(temp_mpx,D)
            tot_dwt += dwt
        new_mpx[-1] = prev_site

    elif mpx1[0].ndim == 3 and mpx2[0].ndim == 4:
        prev_site = einsum('LNR,lNnr->LlnRr',mpx1[0],mpx2[0])
        prev_site = linalg.reshape(prev_site,'ab,c,de')
        for i in range(1,L):
            new_site = einsum('LNR,lNnr->LlnRr',mpx1[i],mpx2[i])
            new_site = linalg.reshape(new_site,'ab,c,de')
            temp_mpx = np.empty(2,dtype=np.object)
            temp_mpx[:] = [prev_site,new_site]
            [new_mpx[i-1],prev_site],dwt = compress(temp_mpx,D)
            tot_dwt += dwt
        new_mpx[-1] = prev_site
            
    elif mpx1[0].ndim == 4 and mpx2[0].ndim == 4:
        prev_site = einsum('LNMR,lMnr->LlNnRr',mpx1[0],mpx2[0])
        prev_site = linalg.reshape(prev_site,'ab,cd,ef')
        for i in range(1,L):
            new_site = einsum('LNMR,lMnr->LlNnRr',mpx1[i],mpx2[i])
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

def vdot(mps1, mps2, direction=0):
    """
    vdot of two MPS, returns scalar

    cf. np.vdot
    """
    return _mps_dot(np.conj(mps1), mps2, direction)

def _mps_dot(mps1, mps2, direction=0):
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
        
    E = einsum('InR, inr -> IiRr', mps1_[0], mps2_[0]) 
    for i in xrange(1, L):
        # contract with bra
        E = einsum('IiRr, RnL -> IirnL', E, mps1_[i])
        # contract with ket
        E = einsum('IirnL, rnl -> IiLl', E, mps2_[i])

    return np.einsum('ijij', E)

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
    return np.sqrt(np.abs(norm_val))

