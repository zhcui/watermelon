import re
import numpy
np = numpy
import scipy
import scipy.sparse
import scipy.sparse.linalg
import sparse
import sparse.coo
COO = sparse.coo.COO

def shape(a):
    return a.shape

def diag(a):
    """
    Perform equivalent of :obj:`numpy.diag`.
    """
    if len(a.shape) == 2:
        return a.to_scipy_sparse().diagonal()
    elif len(a.shape) == 1:
        shape = (a.shape[0], a.shape[0])
        return sparse.COO(coords = np.vstack([a.coords, a.coords]),
                          data = a.data,
                          shape = shape)
    else:
        raise RuntimeError


def diagonal(spmat, axes = None):
    """
    For 2D array, it is equavalent to 'ii -> i'
    For nD array, axes specifies the axes with the same label.
    e.g. shape = (3, 6, 6, 7, 6, 1) with axes = [1, 2, 4]
    would equavalent to 'ijjkjl -> ijkl'

    """
    if axes is None:
        axes = list(range(spmat.ndim))
    if len(axes) < 2:
        return spmat
    mask = (spmat.coords[axes[0]] == spmat.coords[axes[1]])
    for i in xrange(2, len(axes)):
        mask = np.logical_and(mask, (spmat.coords[axes[i]] == spmat.coords[axes[i - 1]]))
    all_axes = np.arange(spmat.ndim)
    indep_axes = []
    flag = True
    for ax in all_axes:
        if ax in axes:
            if flag:
                indep_axes.append(ax)
                flag = False
        else:
            indep_axes.append(ax)

    coords = spmat.coords[indep_axes][:, mask]
    data = spmat.data[mask]
    shape = [spmat.shape[ax] for ax in indep_axes]
    return COO(coords, data, shape=shape, has_duplicates = False)
        
#@profile
def tensordot(a, b, axes=2):
    # taken from sparse.coo
    # only difference is that this fn prevents
    # automatic conversion to ndarray if result is dense
    """
    Perform the equivalent of :obj:`numpy.tensordot`.

    Parameters
    ----------
    a, b : Union[COO, np.ndarray, scipy.sparse.spmatrix]
        The arrays to perform the :code:`tensordot` operation on.
    axes : tuple[Union[int, tuple[int], Union[int, tuple[int]], optional
        The axes to match when performing the sum.

    Returns
    -------
    Union[COO, numpy.ndarray]
        The result of the operation.

    See Also
    --------
    numpy.tensordot : NumPy equivalent function
    """
    # Much of this is stolen from numpy/core/numeric.py::tensordot
    # Please see license at https://github.com/numpy/numpy/blob/master/LICENSE.txt

    try:
        iter(axes)
    except TypeError:
        axes_a = list(range(-axes, 0))
        axes_b = list(range(0, axes))
    else:
        axes_a, axes_b = axes
    try:
        na = len(axes_a)
        axes_a = list(axes_a)
    except TypeError:
        axes_a = [axes_a]
        na = 1
    try:
        nb = len(axes_b)
        axes_b = list(axes_b)
    except TypeError:
        axes_b = [axes_b]
        nb = 1

    # a, b = asarray(a), asarray(b)  # <--- modified
    as_ = a.shape
    nda = a.ndim
    bs = b.shape
    ndb = b.ndim
    equal = True
    if na != nb:
        equal = False
    else:
        for k in range(na):
            if as_[axes_a[k]] != bs[axes_b[k]]:
                equal = False
                break
            if axes_a[k] < 0:
                axes_a[k] += nda
            if axes_b[k] < 0:
                axes_b[k] += ndb
    if not equal:
        raise ValueError("shape-mismatch for sum")

    # Move the axes to sum over to the end of "a"
    # and to the front of "b"
    notin = [k for k in range(nda) if k not in axes_a]
    newaxes_a = notin + axes_a
    N2 = 1
    for axis in axes_a:
        N2 *= as_[axis]
    newshape_a = (-1, N2)
    olda = [as_[axis] for axis in notin]

    notin = [k for k in range(ndb) if k not in axes_b]
    newaxes_b = axes_b + notin
    N2 = 1
    for axis in axes_b:
        N2 *= bs[axis]
    newshape_b = (N2, -1)
    oldb = [bs[axis] for axis in notin]

    at = a.transpose(newaxes_a).reshape(newshape_a)
    bt = b.transpose(newaxes_b).reshape(newshape_b)
    res = _dot(at, bt)
    if isinstance(res, scipy.sparse.spmatrix):
        # this part differs from sparse.coo.tensordot
        res = COO.from_scipy_sparse(res)  # <--- modified
        res.has_duplicates = False
    if isinstance(res, np.matrix):
        res = np.asarray(res)
    return res.reshape(olda + oldb)

#@profile
def dot(a, b):
    # unchanged from sparse.coo
    """
    Perform the equivalent of :obj:`numpy.dot` on two arrays.

    Parameters
    ----------
    a, b : Union[COO, np.ndarray, scipy.sparse.spmatrix]
        The arrays to perform the :code:`dot` operation on.

    Returns
    -------
    Union[COO, numpy.ndarray]
        The result of the operation.

    See Also
    --------
    numpy.dot : NumPy equivalent function.
    COO.dot : Equivalent function for COO objects.
    """
    if not hasattr(a, 'ndim') or not hasattr(b, 'ndim'):
        raise NotImplementedError(
            "Cannot perform dot product on types %s, %s" %
            (type(a), type(b)))

    if a.ndim == 1 and b.ndim == 1:
        return (a * b).sum()

    a_axis = -1
    b_axis = -2 # fixed this??

    if b.ndim == 1:
        b_axis = -1

    return tensordot(a, b, axes=(a_axis, b_axis))

#@profile
def _dot(a, b):
    # unchanged from sparse.coo

    if isinstance(b, COO) and not isinstance(a, COO):
        return _dot(b.T, a.T).T
    aa = a.tocsr()

    if isinstance(b, (COO, scipy.sparse.spmatrix)):
        b = b.tocsc()
        #b = b.tocsr()
    return aa.dot(b)

#@profile
def einsum(idx_str, *tensors, **kwargs):
    # from pyscf.lib.numpy_helper
    '''Perform a more efficient einsum via reshaping to a matrix multiply.

    Current differences compared to numpy.einsum:
    This assumes that each repeated index is actually summed (i.e. no 'i,i->i')
    and appears only twice (i.e. no 'ij,ik,il->jkl'). The output indices must
    be explicitly specified (i.e. 'ij,j->i' and not 'ij,j').
    '''

    DEBUG = kwargs.get('DEBUG', False)

    idx_str = idx_str.replace(' ','')
    indices  = "".join(re.split(',|->',idx_str))
    if '->' not in idx_str or any(indices.count(x)>2 for x in set(indices)):
        return numpy.einsum(idx_str,*tensors)
    if idx_str.count(',') > 1:
        indices  = re.split(',|->',idx_str)
        indices_in = indices[:-1]
        idx_final = indices[-1]
        n_shared_max = 0
        for i in range(len(indices_in)):
            for j in range(i):
                tmp = list(set(indices_in[i]).intersection(indices_in[j]))
                n_shared_indices = len(tmp)
                if n_shared_indices > n_shared_max:
                    n_shared_max = n_shared_indices
                    shared_indices = tmp
                    [a,b] = [i,j]
        tensors = list(tensors)
        A, B = tensors[a], tensors[b]
        idxA, idxB = indices[a], indices[b]
        idx_out = list(idxA+idxB)
        idx_out = "".join([x for x in idx_out if x not in shared_indices])
        C = einsum(idxA+","+idxB+"->"+idx_out, A, B)
        indices_in.pop(a)
        indices_in.pop(b)
        indices_in.append(idx_out)
        tensors.pop(a)
        tensors.pop(b)
        tensors.append(C)
        return einsum(",".join(indices_in)+"->"+idx_final,*tensors)

    A, B = tensors
    # Call numpy.asarray because A or B may be HDF5 Datasets 
    # A = numpy.asarray(A, order='A')
    # B = numpy.asarray(B, order='A')
    # if A.size < 2000 or B.size < 2000:
    #     return numpy.einsum(idx_str, *tensors)

    # Split the strings into a list of idx char's
    idxA, idxBC = idx_str.split(',')
    idxB, idxC = idxBC.split('->')
    idxA, idxB, idxC = [list(x) for x in [idxA,idxB,idxC]]
    assert(len(idxA) == A.ndim)
    assert(len(idxB) == B.ndim)

    if DEBUG:
        print("*** Einsum for", idx_str)
        print(" idxA =", idxA)
        print(" idxB =", idxB)
        print(" idxC =", idxC)

    # Get the range for each index and put it in a dictionary
    rangeA = dict()
    rangeB = dict()
    #rangeC = dict()
    for idx,rnge in zip(idxA,A.shape):
        rangeA[idx] = rnge
    for idx,rnge in zip(idxB,B.shape):
        rangeB[idx] = rnge
    #for idx,rnge in zip(idxC,C.shape):
    #    rangeC[idx] = rnge

    if DEBUG:
        print("rangeA =", rangeA)
        print("rangeB =", rangeB)

    # Find the shared indices being summed over
    shared_idxAB = list(set(idxA).intersection(idxB))
    #if len(shared_idxAB) == 0:
    #    return np.einsum(idx_str,A,B)
    idxAt = list(idxA)
    idxBt = list(idxB)
    inner_shape = 1
    insert_B_loc = 0
    for n in shared_idxAB:
        if rangeA[n] != rangeB[n]:
            err = ('ERROR: In index string %s, the range of index %s is '
                   'different in A (%d) and B (%d)' %
                   (idx_str, n, rangeA[n], rangeB[n]))
            raise RuntimeError(err)

        # Bring idx all the way to the right for A
        # and to the left (but preserve order) for B
        idxA_n = idxAt.index(n)
        idxAt.insert(len(idxAt)-1, idxAt.pop(idxA_n))

        idxB_n = idxBt.index(n)
        idxBt.insert(insert_B_loc, idxBt.pop(idxB_n))
        insert_B_loc += 1

        inner_shape *= rangeA[n]

    if DEBUG:
        print("shared_idxAB =", shared_idxAB)
        print("inner_shape =", inner_shape)

    # Transpose the tensors into the proper order and reshape into matrices
    new_orderA = [idxA.index(idx) for idx in idxAt]
    new_orderB = [idxB.index(idx) for idx in idxBt]

    if DEBUG:
        print("Transposing A as", new_orderA)
        print("Transposing B as", new_orderB)
        print("Reshaping A as (-1,", inner_shape, ")")
        print("Reshaping B as (", inner_shape, ",-1)")

    shapeCt = list()
    idxCt = list()
    for idx in idxAt:
        if idx in shared_idxAB:
            break
        shapeCt.append(rangeA[idx])
        idxCt.append(idx)
    for idx in idxBt:
        if idx in shared_idxAB:
            continue
        shapeCt.append(rangeB[idx])
        idxCt.append(idx)
    new_orderCt = [idxCt.index(idx) for idx in idxC]

    if A.size == 0 or B.size == 0:
        shapeCt = [shapeCt[i] for i in new_orderCt]
        return numpy.zeros(shapeCt, dtype=numpy.result_type(A,B))

    At = A.transpose(new_orderA)
    Bt = B.transpose(new_orderB)

    # if At.flags.f_contiguous:
    At = At.reshape((-1,inner_shape))
    # else:
    #     At = numpy.asarray(At.reshape(-1,inner_shape), order='C')
    # if Bt.flags.f_contiguous:
    Bt = Bt.reshape((inner_shape,-1))
    # else:
    #     Bt = numpy.asarray(Bt.reshape(inner_shape,-1), order='C')
    result = dot(At,Bt).reshape(shapeCt).transpose(new_orderCt)
    return result

#def svd(idx, a, D=0, preserve_uv=None):
#    idx0 = re.split(",", idx)
#    assert len(idx0) == 2
#    idx0[0].replace(" ", "")
#
#    nsplit = len(idx0[0]) 
#
#    a_shape = a.shape
#    a = a.reshape([np.prod(a.shape[:nsplit]), -1])
#
#    M = min(a.shape[0], a.shape[1])
#    if D > 0:
#        M = min(D, M)
#
#    # Arnoldi based, special M==N exception
#    if M == min(a.shape[0], a.shape[1]):
#        print "Using Dense SVD"
#        u, s, vt = scipy.linalg.svd(a.todense(), full_matrices=False)
#    else:
#        print "Using Sparse SVD"
#        u, s, vt = scipy.sparse.linalg.svds(a.to_scipy_sparse(), M)
#
#    # print "Singular values are", s
#    # print u.shape
#    # print s.shape
#    # print vt.shape
#    u = sparse.coo.COO.from_numpy(u)
#    s = sparse.coo.COO.from_numpy(s)
#    vt = sparse.coo.COO.from_numpy(vt)
#
#    dwt = None
#
#    # Need to debug this stuff below
#    # if preserve_uv == "u":
#    #     ubig = sparse.coo.COO(coords = u.coords, data = u.data,
#    #                           shape = a.shape)
#    #     sbig = sparse.coo.COO(coords = s.coords, data = s.data,
#    #                           shape = (a.shape[1],))
#    #     vtbig = sparse.coo.COO(coords = vt.coords, data = vt.data,
#    #                            shape = (a.shape[1], a.shape[1]))
#    #     u, s, vt = ubig, sbig, vtbig 
#    # elif preserve_uv == "v":
#    #     vtbig = sparse.coo.COO(coords = v.coords, data = v.data,
#    #                            shape = a.shape)
#    #     sbig = sparse.coo.COO(coords = s.coords, data = s.data,
#    #                           shape = (a.shape[0],))
#    #     ubig = sparse.coo.COO(coords = u.coords, data = u.data,
#    #                           shape = (a.shape[0], a.shape[0]))
#    #     u, s, vt = ubig, sbig, vtbig 
#
#    # np.set_printoptions(precision=3)
#    # print "ortho"
#    # ud = u.todense()
#    # vtd = vt.todense()
#    # u1 = np.dot(u.todense().T, u.todense())
#    # vt1= np.dot(vt.todense(), vt.todense().T)
#
#    # print u1
#    # print vt1
#    # print "deviations from 1"
#    # print np.linalg.norm(u1-np.eye(ud.shape[1]))
#    # print np.linalg.norm(vt1-np.eye(vtd.shape[0]))
#    
#    u = u.reshape((a_shape[:nsplit] + (-1,)))
#    vt = vt.reshape(((-1,) + a_shape[nsplit:]))
#
#    # print "all my stuff"
#    # print u
#    # print s
#    # print "----------"
#    # print u.todense(), u.shape
#    # print s.todense(), s.shape
#    # print vt.todense(), vt.shape
#    # print np.dot(u.todense(), np.dot(np.diag(s.todense()), vt.todense()))
#    # print a.todense()
#    # print "----------"
#    return u, s, vt, dwt
#
#def _svd():
#    pass

def svd(idx, a, D=0, preserve_uv=None):
    #  ZHC NOTE This is a svd using clustering dense svd.
    from sparse.coo.core import block_svd as svd
    idx0 = re.split(",", idx)
    assert len(idx0) == 2
    idx0[0].replace(" ", "")

    nsplit = len(idx0[0]) 

    a_shape = a.shape
    a = a.reshape((np.prod(a.shape[:nsplit]), -1))

    #u, s, vt = scipy.linalg.svd(a, full_matrices = False)

    M = min(a.shape[0], a.shape[1])
    if D > 0:
        M = min(D, M)

    if preserve_uv is None:
        u, s, vt, dwt = svd(a, sort = True, full_matrices = False, dim_keep = M, return_dwt = True)
    else:
        # TODO take care of truncation
        u, s, vt, dwt = svd(a, sort = True, full_matrices = True, return_dwt = True)

    u = u.reshape(a_shape[:nsplit] + (-1,))
    vt = vt.reshape((-1,) + a_shape[nsplit:])

    return u, s, vt, dwt

