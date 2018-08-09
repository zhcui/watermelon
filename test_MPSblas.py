#! /usr/bin/env python
import numpy as np
import MPSblas
import pytest

def test_rand():
    dp = [1, 5, 4, 2, 4]
    mps = MPSblas.rand(dp, 7)
    mps_res_shape = [(1, 1, 1),
                     (1, 5, 5),
                     (5, 4, 7),
                     (7, 2, 4),
                     (4, 4, 1)]
    
    print "mps"
    for i, x in enumerate(mps):
        print x.shape
        assert(np.allclose(x.shape, mps_res_shape[i]))
    
    dp = [(1,2), (5,3), (4,4)]
    mpo = MPSblas.rand(dp, 4)
    mpo_res_shape = [(1, 1, 2, 2),
                     (2, 5, 3, 4),
                     (4, 4, 4, 1)]
        
    print "mpo"
    for i, x in enumerate(mpo):
        print x.shape
        assert(np.allclose(x.shape, mpo_res_shape[i]))


def test_calc_dim():
    dps = [5, 1, 4]
    drs = MPSblas.calc_dim(dps, None)
    print drs

    drs = MPSblas.calc_dim(dps, D = 3)
    print drs


def test_product_state():
    dp = (4, 5, 3)
    try:
        mps = MPSblas.product_state(dp, [5, 5, 4])
    except:
        print "ok"

    mps = MPSblas.product_state(dp, [0, 1, 2])

    print MPSblas.element(mps, [0, 1, 2])
    for occ in np.ndindex(dp):
        print occ, MPSblas.element(mps, tuple(occ))
    print mps

def test_asfull():
    dp = (4, 5, 3)
    mps = MPSblas.product_state(dp, [0, 1, 2])

    
    
    
def test_scal():
    dps = [1, 5, 4]
    mps1 = MPSblas.rand(dps, 4)

    alpha = -1
    mps2 = MPSblas.scal(alpha, mps1)
  
    norm1 = MPSblas.norm(mps1)
    norm2 = MPSblas.norm(mps2)
    ovlp  = MPSblas.dot(mps1.conj(),mps2)

    print 'scaling used: ', np.abs(alpha), np.angle(alpha)
    print 'ratio of norms: ', norm2/norm1
    print 'scaling', ovlp/(norm1**2)


def test_axpby():
    dps = [1,4,5]
    mps1 = MPSblas.rand(dps,4)
    mps2 = MPSblas.rand(dps,3)
 
    alpha = 1
    beta  = -1+3j
    out_mps = MPSblas.axpby(alpha,mps1,beta,mps2)

    for s in out_mps:  print s.shape

    print MPSblas.norm(mps1), MPSblas.norm(mps2), MPSblas.dot(mps1,mps2)
    print 'expect', MPSblas.norm(mps1)**2*np.abs(alpha)**2 + MPSblas.norm(mps2)**2*np.abs(beta)**2\
                  + MPSblas.dot(mps2,mps1.conj())*np.conj(alpha)*beta\
                  + MPSblas.dot(mps1,mps2.conj())*alpha*np.conj(beta)
    print MPSblas.norm(out_mps)**2

    
def test_compress():
    dps = [4, 4, 2, 3]

    mps1 = MPSblas.rand(dps, D=5)

    mps2, dwt2 = MPSblas.compress(mps1, D=3)
    mps3, dwt3 = MPSblas.compress(mps1, D=2)
    mps4, dwt4 = MPSblas.compress(mps1, D=1)
    mps5, dwt5 = MPSblas.compress(mps1, D=2, direction=1)
    mps6, dwt6 = MPSblas.compress(mps1, D=1, direction=1)
    print dwt2, dwt3, dwt4, dwt5, dwt6

    print MPSblas.dot(mps1, mps1)
    print MPSblas.dot(mps1, mps2)
    print MPSblas.dot(mps1, mps3)
    print MPSblas.dot(mps1, mps4)
    print MPSblas.dot(mps1, mps5)
    print MPSblas.dot(mps1, mps6)

    
    
