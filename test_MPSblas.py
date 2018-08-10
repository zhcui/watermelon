#! /usr/bin/env python
import numpy as np
import MPSblas
from MPSblas import asfull
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
    assert(np.allclose(drs, [4, 4]))

    drs = MPSblas.calc_dim(dps, D = 3)
    print drs
    assert(np.allclose(drs, [3, 3]))


def test_product_state():
    dp = (4, 5, 3)

    # out of range case
    occ_idx = (5, 5, 4)
    with pytest.raises(IndexError):
        mps = MPSblas.product_state(dp, occ_idx)

    occ_idx = (0, 1, 2)
    mps = MPSblas.product_state(dp, occ_idx)

    for occ in np.ndindex(dp):
        if np.allclose(occ, occ_idx):
            assert (np.allclose(MPSblas.element(mps, occ), 1.0))
        else:
            assert (np.allclose(MPSblas.element(mps, occ), 0.0))


def test_asfull():
    dp = (4, 5, 3)

    # test elemwise
    mps = MPSblas.product_state(dp, [0, 1, 2])
    vec = MPSblas.asfull(mps)

    for i, occ in enumerate(np.ndindex(dp)):
        assert abs(MPSblas.element(mps, occ) - vec[i]) < 1.e-10

    # test mpo x mpo 
    mpo = MPSblas.rand(zip(dp, dp))
    mat = MPSblas.asfull(mpo)
    
    mpo2 = MPSblas.dot(mpo, mpo)
    mat2 = np.dot(mat, mat)

    print np.linalg.norm(mat2)
    print np.linalg.norm(MPSblas.asfull(mpo2))
    
    assert np.linalg.norm(mat2 - MPSblas.asfull(mpo2)) < 1.e-9

    # test mpo x mps
    mps = MPSblas.rand(dp)
    vec = MPSblas.asfull(mps)
    matvec = np.dot(mat, vec)
    mps1 = MPSblas.dot(mpo, mps)

    assert np.linalg.norm(matvec - MPSblas.asfull(mps1)) < 1.e-9

    # test mps x mpo
    mps1 = MPSblas.dot(mps, mpo)
    vecmat = np.dot(vec, mat)
    assert np.linalg.norm(vecmat - MPSblas.asfull(mps1)) < 1.e-9
    
    # test mps x mps
    norm1 = MPSblas.dot(mps, mps)
    norm = np.dot(vec, vec)
    assert abs(norm - norm1) < 1.e-9


    
def test_mul():
    dps = [1, 5, 4]
    mps1 = MPSblas.rand(dps, 4)

    alpha = -1
    mps2 = MPSblas.mul(alpha, mps1)
  
    norm1 = MPSblas.norm(mps1)
    norm2 = MPSblas.norm(mps2)
    ovlp  = MPSblas.dot(mps1.conj(),mps2)

    print 'scaling used: ', np.abs(alpha), np.angle(alpha)
    print 'ratio of norms: ', norm2 / norm1
    print 'scaling', ovlp / (norm1**2)
    assert(np.allclose(norm1, norm2))
    assert(np.allclose(ovlp / (norm1 ** 2), alpha))


def test_axpby():
    dps = [1, 4, 5]
    mps1 = MPSblas.rand(dps, 4)
    mps2 = MPSblas.rand(dps, 3)
 
    alpha = 1
    beta  = -1 + 3j
    out_mps = MPSblas.axpby(alpha, mps1, beta, mps2)

    for s in out_mps:  print s.shape

    print MPSblas.norm(mps1), MPSblas.norm(mps2), MPSblas.dot(mps1,mps2)
    expect_norm = MPSblas.norm(mps1)**2*np.abs(alpha)**2 + MPSblas.norm(mps2)**2*np.abs(beta)**2\
                  + MPSblas.dot(mps2,mps1.conj())*np.conj(alpha)*beta\
                  + MPSblas.dot(mps1,mps2.conj())*alpha*np.conj(beta)
    print expect_norm 
    print MPSblas.norm(out_mps)**2
    assert(np.allclose(MPSblas.norm(out_mps)**2, expect_norm))

    mps1_f = asfull(mps1)
    mps2_f = asfull(mps2)
    out_mps_f = asfull(out_mps)
    out_mps_f_standard = alpha * mps1_f + beta * mps2_f
    
    assert(np.allclose(out_mps_f, out_mps_f_standard))


    
def test_compress():
    dps = [4, 4, 2, 3]

    mps1 = MPSblas.rand(dps, D = 5)

    mps2, dwt2 = MPSblas.compress(mps1, D = 3)
    mps3, dwt3 = MPSblas.compress(mps1, D = 2)
    mps4, dwt4 = MPSblas.compress(mps1, D = 1)
    mps5, dwt5 = MPSblas.compress(mps1, D = 2, direction = 1)
    mps6, dwt6 = MPSblas.compress(mps1, D = 1, direction = 1)
    print dwt2, dwt3, dwt4, dwt5, dwt6

    print MPSblas.dot(mps1, mps1)
    print MPSblas.dot(mps1, mps2)
    print MPSblas.dot(mps1, mps3)
    print MPSblas.dot(mps1, mps4)
    print MPSblas.dot(mps1, mps5)
    print MPSblas.dot(mps1, mps6)

   
def test_flatten():
    dp = [(1,2), (5,3), (4,4)]
    mpos = MPSblas.rand(dp, 4)
    mps_out = MPSblas.flatten(mpos)
    for i in xrange(len(mps_out)):
        assert(mps_out[i].ndim == 3)
    
def test_dot():
    dps = [1,4,4,2]
    mps1 = MPSblas.rand(dps,4)
    mps2 = MPSblas.rand(dps,3)

    dps_o = [(d,d) for d in dps]
    mpo1 = MPSblas.rand(dps_o,4)
    mpo2 = MPSblas.rand(dps_o,3)

    sh_s1 = [m.shape for m in mps1]
    sh_s2 = [m.shape for m in mps2]
    sh_o1 = [m.shape for m in mpo1]
    sh_o2 = [m.shape for m in mpo2]

    ## <mps1|mps2>
    ss12 = MPSblas.dot(mps1,mps1)
    ## mpo1|mps1>
    os11 = MPSblas.dot(mpo1,mps1)
    ## mpo2*mpo1
    oo21 = MPSblas.dot(mpo2,mpo1)
    ## <mps1|mpo1
    so11 = MPSblas.dot(mps1,mpo1)

    assert np.isscalar(ss12), '<mps|mps> should return scalar'
    # print [(sh_o1[i][0]*sh_s1[i][0],sh_s1[i][1],sh_o1[i][2]*sh_s1[i][2]) for i in range(len(dps))]
    # print [o.shape for o in os11]
    assert np.all([os11[i].shape == (sh_o1[i][0]*sh_s1[i][0],sh_s1[i][1],sh_o1[i][3]*sh_s1[i][2]) for i in range(len(dps))])
    assert np.all([oo21[i].shape == (sh_o2[i][0]*sh_o1[i][0],sh_o2[i][1],sh_o1[i][2],sh_o2[i][3]*sh_o1[i][3]) for i in range(len(dps))])
    assert np.all([so11[i].shape == (sh_s1[i][0]*sh_o1[i][0],sh_s1[i][1],sh_s1[i][2]*sh_o1[i][3]) for i in range(len(dps))])

    # for m in os11:  print m.shape
    # for m in oo21:  print m.shape
    # for m in so11:  print m.shape 

    dps = [1,3,4,1]
    mps3 = MPSblas.rand(dps,4)
    dps_o = [(d,d) for d in dps]
    mpo3 = MPSblas.rand(dps_o,3)

    try:
        ss31 = MPSblas.dot(mps3,mps1)
    except:
        print 'error, d mismatch'
    
    try:
        so31 = MPSblas.dot(mps3,mpo1)
    except:
        print 'error, d mismatch'

    try:
        oo31 = MPSblas.dot(mpo3,mpo1)
    except:
        print 'error, d mismatch'

 
    
