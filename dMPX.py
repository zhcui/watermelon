import numpy as np
import gMPX

"""
MPX with dense tensors
"""  
def empty(dp, D = None, bc = None, dtype=np.float64):
    return gMPX.create(dp, D, bc, fn=np.empty, dtype=dtype)

def zeros(dp, D = None, bc = None, dtype=np.float64):
    return gMPX.create(dp, D, bc, fn=np.zeros, dtype=dtype)

def rand(dp, D = None, bc = None, seed = None, dtype=np.float64):
    if dtype != np.float64:
        raise NotImplementedError

    if seed is not None:
        np.random.seed(seed)
    def fn(a, dtype):
        return np.array(np.random.random(a), dtype=dtype)
    return gMPX.create(dp, D, bc, fn=fn, dtype=dtype)

def product_state(dp, occ, D=None, bc=None):
    """
    Parameters
    ----------
    dp : sequence of int
      Physical dimension of MPX
    D : int 
      max bond (matrices contain one non-zero element)
    occ : sequence of int (MPS) / tuple[2] (MPO)
      non-zero physical index in state
    
    Returns
    -------
    mps : MPS product state according to occ
    """
    L = len(dp)
    mps = zeros(dp, D, bc)
    for i in range(L):
        mps[i][0, occ[i], 0] = 1.

    return mps

def add(mpx1, mpx2):
    """
    Add MPX1 to MPX2

    Parameters
    ----------
    mpx1, mpx2 : MPS or MPO
    bc : "obc" or "pbc"
      if "obc" first and last tensors
      are collapsed to shape (1,...), (...,1)

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
            new_site[:l1,:,:r1] = mpx1[i].reshape((l1,n1,r1))
            new_site[:l2,:,r1:] = mpx2[i].reshape((l2,n2,r2))
        elif i==L-1:
            new_site = np.zeros((l1+l2,n1,max(r1,r2)),dtype=dtype)
            new_site[:l1,:,:r1] = mpx1[i].reshape((l1,n1,r1))
            new_site[l1:,:,:r2] = mpx2[i].reshape((l2,n2,r2))
        else:
            new_site = np.zeros((l1+l2,n1,r1+r2),dtype=dtype)
            new_site[:l1,:,:r1] = mpx1[i].reshape((l1,n1,r1))
            new_site[l1:,:,r1:] = mpx2[i].reshape((l2,n2,r2))

        nsh = new_site.shape
        new_site = new_site.reshape((nsh[0],)+sh1[1:-1]+(nsh[-1],))
        new_mpx[i] = new_site
    
    return new_mpx

def overwrite(mpx, out=None):
    """
    Overwrites tensors of mpx2 with tensors of mpx1,
    with fixed shape of mpx2 tensors.

    Parameters
    ----------
    mpx : MPX (source)
    out : MPX (target) [modified]
    """
    for m1, m2 in zip(out, mpx):
        if len(m2.shape) == 3:
            m1[:m2.shape[0],:m2.shape[1],:m2.shape[2]] = m2[:,:,:]
        else:
            assert len(m2.shape) == 4
            m1[:m2.shape[0],:m2.shape[1],:m2.shape[2],:m2.shape[3]] = m2[:,:,:,:]
