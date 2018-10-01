import numpy as np
import sparse as sp
import sparse.coo
import gMPX as MPX

"""
Sparse MPX functions
"""
def zeros(dp, D = None, bc = None, dtype = np.float64, fix_D = False):
    def _from_np_zeros(shape, dtype):
        return sp.COO.from_numpy(np.zeros(shape, dtype))
    return MPX.create(dp, D, bc, fn=_from_np_zeros, fix_D = fix_D)

def rand(dp, D = None, bc = None, seed = None, density = 0.8, dtype=np.float64, fix_D = False):
    # random float, upcast to dtype
    if seed is not None:
        np.random.seed(seed)
    def fn(shape, dtype):
        return sp.random(shape, density=density)

    if dtype != np.float64:
        raise NotImplementedError
    
    return MPX.create(dp, D, bc, fn=fn, dtype=None, fix_D = fix_D)

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
    bc : "obc" or "pbc"
      if "obc" first and last tensors
      are collapsed to shape (1,...), (...,1)

    Returns
    -------
    new_mpx : sum of MPS
    """
    L = len(mpx1)

    new_mpx = np.empty(L, dtype=np.object)

    assert len(mpx1)==len(mpx2), 'need to have same lengths: (%d,%d)' % (len(mpx1), len(mpx2))
    
    for i in range(L):
        sh1 = mpx1[i].shape
        sh2 = mpx2[i].shape
        assert sh1[1: -1] == sh2[1: -1], 'need physical bonds at site %d to match' % (i)        

        l1,n1,r1 = sh1[0],np.prod(sh1[1:-1]),sh1[-1]
        l2,n2,r2 = sh2[0],np.prod(sh2[1:-1]),sh2[-1]
        
        crds1 = mpx1[i].coords.copy()
        crds2 = mpx2[i].coords.copy()
        if i==0:
            crds2[-1] += r1
            nsh = (max(l1,l2),) + sh1[1:-1] + (r1+r2,)
        elif i==L-1:
            crds2[0] += l1
            nsh = (l1+l2,) + sh1[1:-1] + (max(r1,r2),)
        else:
            crds2[0] += l1
            crds2[-1] += r1
            nsh = (l1+l2,) + sh1[1:-1] + (r1+r2,)
            
        new_crds = np.hstack([crds1, crds2])
        new_data = np.hstack([mpx1[i].data, mpx2[i].data])
        
        new_mpx[i] = sp.COO(new_crds, new_data, shape = nsh)
        shape  = new_mpx[i].shape

    return new_mpx

