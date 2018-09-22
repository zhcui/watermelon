#! /usr/bin/env python

import numpy as np
from ctypes import *
import scipy.sparse as spsp
import scipy.sparse._sparsetools as _sparsetools
import sparse
from sparse import COO

try:
    libspmm = cdll.LoadLibrary("../libspmm.so")
    FOUND_LIB = True
except (OSError):
    FOUND_LIB = False

if FOUND_LIB:
    def get_indptr(A):
        #row = A.row
        row = A.coords[0]
        indptr = np.zeros(A.shape[0] + 1, dtype=np.int32)
        np.cumsum(np.bincount(row, minlength=A.shape[0]), out=indptr[1:])
        return indptr
    #@profile
    def mul_coo_coo(A, B):

        M, K = A.shape
        K_, N = B.shape
        assert(K == K_)

        # convert dtype to c_int and c_double
        rowIndex_A = get_indptr(A)
        #columns_A = A.col.astype(np.int32)
        columns_A = A.coords[1].astype(np.int32)
        values_A = A.data.astype(np.double)
        
        rowIndex_B = get_indptr(B)
        #columns_B = B.col.astype(np.int32)
        columns_B = B.coords[1].astype(np.int32)
        values_B = B.data.astype(np.double)

        # output variables
        pointerB_C_p = POINTER(c_int)()
        pointerE_C_p = POINTER(c_int)()
        columns_C_p = POINTER(c_int)()
        values_C_p = POINTER(c_double)()
        nnz = c_int(0)
        handle_C = c_void_p() # used to free data

        # calculation
        libspmm.spmm(byref(c_int(M)), byref(c_int(N)), byref(c_int(K)), \
                rowIndex_A.ctypes.data_as(c_void_p), columns_A.ctypes.data_as(c_void_p), values_A.ctypes.data_as(c_void_p), \
                rowIndex_B.ctypes.data_as(c_void_p), columns_B.ctypes.data_as(c_void_p), values_B.ctypes.data_as(c_void_p), \
                byref(pointerB_C_p), byref(pointerE_C_p), byref(columns_C_p), byref(values_C_p), byref(nnz), byref(handle_C))
        
        nnz = nnz.value

        # convert to numpy object
        buffer_tmp = np.core.multiarray.int_asbuffer(addressof(values_C_p.contents),\
                np.dtype(np.double).itemsize * nnz)
        values_C = np.frombuffer(buffer_tmp, np.double).copy()

        buffer_tmp = np.core.multiarray.int_asbuffer(
                    addressof(columns_C_p.contents), np.dtype(np.int32).itemsize * nnz)
        columns_C = np.frombuffer(buffer_tmp, np.int32).copy()

        buffer_tmp = np.core.multiarray.int_asbuffer(
                    addressof(pointerB_C_p.contents), np.dtype(np.int32).itemsize * M)
        pointerB_C = np.frombuffer(buffer_tmp, np.int32).copy()
        pointerB_C = np.append(pointerB_C, np.array(nnz, dtype = np.int32)) # automatically do the copy

        # free C
        libspmm.free_handle(byref(handle_C))

        # compute full row of C
        rows_C = np.empty(nnz, dtype = columns_C.dtype)
        _sparsetools.expandptr(M, pointerB_C, rows_C)
       
        if nnz > 1000000:
            # sort the cols and data
            idx_C = np.empty(nnz, dtype = np.int64)
            pre_num = 0
            for i in xrange(M):
                idx_C[pointerB_C[i]:pointerB_C[i + 1]] = np.argsort(columns_C[pointerB_C[i]:pointerB_C[i + 1]]) + pre_num
                pre_num += pointerB_C[i + 1] - pointerB_C[i]
            columns_C = columns_C[idx_C]
            values_C = values_C[idx_C]
            
            return COO(np.vstack((rows_C, columns_C)), data = values_C, shape = (M, N), sorted = True, has_duplicates = False)
        else:
            return COO(np.vstack((rows_C, columns_C)), data = values_C, shape = (M, N), sorted = False, has_duplicates = False)

else:
    mul_coo_coo = sparse.common.dot

if __name__ == '__main__':
   
    import time
    M = 2500
    N = 10
    K = 10
    A_d = np.random.random((M, K))
    A_d[A_d < 0.9] = 0.0
    B_d = np.random.random((K, N))
    B_d[B_d < 0.9] = 0.0
    
    t1 = time.time()
    C_d =  A_d.dot(B_d) # dense reference
    t2 = time.time()
    
    print "a matrix"
    print A_d
    print "b matrix"
    print B_d
    print "c reference"
    print C_d

    #A = spsp.coo_matrix(A_d)
    #B = spsp.coo_matrix(B_d)
    A = COO(A_d)
    B = COO(B_d)

    
    t3 = time.time()
    C = A.dot(B)
    t4 = time.time()

    t5 = time.time()
    C_coo = mul_coo_coo(A, B)
    t6 = time.time()
    
    print "MKL sparse" 
    
    #C = spsp.csr_matrix((data, col, indptr), shape = (M, N))
    print "c calculated from spmm"
    print C_coo.todense()

    assert(np.allclose(C_d, C_coo.todense()))
    print "dense time"
    print t2 - t1
    print "COO time"
    print t4 - t3
    print "MKL sparse time"
    print t6 - t5
