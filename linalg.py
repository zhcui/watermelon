import re
import numpy as np
import scipy, scipy.linalg

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

    nsplit = len(idx0) 

    a_shape = a.shape
    a = np.reshape(a, [np.prod(a.shape[:nsplit]), -1])
    u, s, vt = scipy.linalg.svd(a, full_matrices = False)
    
    M = len(s)
    if DMAX > 0:
        M = min(DMAX, M)

    u = u[:,:M]
    s = s[:M]
    vt = vt[:M,:]

    dwt = np.sum(s[M:])
    
    u = np.reshape(u, (a_shape[:nsplit] + (-1,)))
    vt = np.reshape(vt, ((-1,) + a_shape[nsplit:]))
    return u, s, vt, dwt
