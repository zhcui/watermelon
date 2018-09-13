import re
import numpy as np
import scipy, scipy.linalg

def reshape(a, idx):
    """ 
    Reshape tensors
    
   
    idx: subscripts to split according to ','
         '...' means reshape(-1) if at the beginning/end
               means leave untouched if in the middle
    a:   ndarray to reshape
    
    Returns
    -------
    new_a:  reshaped ndarray

    """
    idx0 = re.split(",", idx)
    ellipse = [x == '...' for x in idx0]
    splits  = [len(x) for x in idx0]
    L = len(splits)
    
    a_sh   = a.shape
    new_sh = []

    indL = 0
    for i in range(L):
        indR = indL + splits[i]
        if ellipse[i]:
            if i==0 or i==L-1:  
                new_sh += [-1]
            else:
                indR = L-np.sum(splits[i+1:])
                new_sh += [s for s in a_sh[indL:indR]]
        else:
            new_sh += [np.prod(a_sh[indL:indR],dtype=np.int)]
        indL = indR  
    
    return a.reshape(new_sh)



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
        singular value
    vt : ndarray
        right matrix
    dwt: float
        discarded wt
    """
    idx0 = re.split(",", idx)
    assert len(idx0) == 2
    idx0[0].replace(" ", "")

    nsplit = len(idx0[0]) 

    a_shape = a.shape
    a = np.reshape(a, [np.prod(a.shape[:nsplit]), -1])
    u, s, vt = scipy.linalg.svd(a, full_matrices = True)
    

    M = len(s)
    if DMAX > 0:
        M = min(DMAX, M)

    dwt = np.sum(s[M:])
    u = u[:,:M]
    s = s[:M]
    vt = vt[:M,:] # ZHC NOTE How to truncate in BCOO version?

    u = np.reshape(u, (a_shape[:nsplit] + (-1,)))
    vt = np.reshape(vt, ((-1,) + a_shape[nsplit:]))
    return u, s, vt, dwt

def test_linalg():
    import numpy.random
    
    a = np.random.random([4,4,4])
    print a.shape
    u, s, vt, dwt = svd("ij,k", a, 2)
    print u.shape, s.shape, vt.shape, dwt
    a2 = np.einsum("ijk,kl,lr", u, np.diag(s), vt)
    print np.linalg.norm(a2-a)

    u, s, vt, dwt = svd("i,jk", a, 2)
    print u.shape, s.shape, vt.shape, dwt
    a2 = np.einsum("ij,jk,klr", u, np.diag(s), vt)
    print np.linalg.norm(a2-a)
    
    
