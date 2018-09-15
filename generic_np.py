import numpy as np
import sparse
import sparse.coo
import numpy_helper
import sparse_helper

def reshape(a, idx):
    """ 
    Reshape tensors with index notation
    
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

def dot(a, b):
    if isinstance(a, sparse.coo.COO) or isinstance(b, sparse.coo.COO):
        return sparse_helper.dot(a, b)
    else:
        return np.dot(a, b)
    
def einsum(idx, *tensors, **kwargs):
    if any(isinstance(a, sparse.coo.COO) for a in tensors):
        return sparse_helper.einsum(idx, *tensors)
    else:
        return np.einsum(idx, *tensors, **kwargs)

def diag(a):
    if isinstance(a, sparse.coo.COO):
        return sparse_helper.diag(a)
    else:
        return np.diag(a)
    
def svd(idx, a, D=0, preserve_uv=None):
    """
    Thin Singular Value Decomposition

    idx : subscripts to split 
    a : ndarray
      matrix to be decomposed.
    D : int
      max #singular values to keep
    preserve_uv : 'u', 'v', None
      if 'u' or 'v', 'u', 'v' have same shape as a
      with extra entries in u, s, v filled with zero
     
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
    if isinstance(a, np.ndarray):
        return numpy_helper.svd(idx, a, D, preserve_uv)
    elif isinstance(a, sparse.coo.COO):
        return sparse_helper.svd(idx, a, D, preserve_uv)
    else:
        raise RuntimeError
