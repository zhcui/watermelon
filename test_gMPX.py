#! /usr/bin/env python
import numpy as np
import gMPX as MPX
import dMPX
import pytest

def test_rand():
    dp = [1, 5, 4, 2, 4]
    mps = dMPX.rand(dp, 7, bc="obc")
    mps_res_shape = [(1, 1, 1),
                     (1, 5, 5),
                     (5, 4, 7),
                     (7, 2, 4),
                     (4, 4, 1)]
    for i, x in enumerate(mps):
        assert(np.allclose(x.shape, mps_res_shape[i]))

    dp = [1, 5, 4, 2, 4]
    mps = dMPX.rand(dp, 7, bc="pbc")
    mps_res_shape = [(7, 1, 7),
                     (7, 5, 7),
                     (7, 4, 7),
                     (7, 2, 7),
                     (7, 4, 7)]
    
    for i, x in enumerate(mps):
        assert(np.allclose(x.shape, mps_res_shape[i]))
    
    print "mps"
    
    dp = [(1,2), (5,3), (4,4)]
    mpo = dMPX.rand(dp, 4, bc="obc")
    mpo_res_shape = [(1, 1, 2, 2),
                     (2, 5, 3, 4),
                     (4, 4, 4, 1)]
    
    print "mpo"
    for i, x in enumerate(mpo):
        print x.shape
        assert(np.allclose(x.shape, mpo_res_shape[i]))

def test_obc_dim():
    dps = [5, 1, 4]
    drs = MPX.obc_dim(dps, None)
    print drs
    assert(np.allclose(drs, [4, 4]))

    drs = MPX.obc_dim(dps, D = 3)
    print drs
    assert(np.allclose(drs, [3, 3]))


def test_product_state():
    dp = (4, 5, 3)

    # out of range case
    occ_idx = (5, 5, 4)
    with pytest.raises(IndexError):
        mps = dMPX.product_state(dp, occ_idx, D=1, bc="obc")
    with pytest.raises(IndexError):
        mps = dMPX.product_state(dp, occ_idx, D=1, bc="pbc")

    occ_idx = (0, 1, 2)
    mps = dMPX.product_state(dp, occ_idx, D=1, bc="obc")
    for occ in np.ndindex(dp):
        if np.allclose(occ, occ_idx):
            assert (np.allclose(MPX.element(mps, occ, bc="obc"), 1.0))
        else:
            assert (np.allclose(MPX.element(mps, occ, bc="obc"), 0.0))

    occ_idx = (0, 1, 2)
    mps = dMPX.product_state(dp, occ_idx, D=4, bc="obc")
    for occ in np.ndindex(dp):
        if np.allclose(occ, occ_idx):
            assert (np.allclose(MPX.element(mps, occ, bc="obc"), 1.0))
        else:
            assert (np.allclose(MPX.element(mps, occ, bc="obc"), 0.0))

    occ_idx = (0, 1, 2)
    mps = dMPX.product_state(dp, occ_idx, D=4, bc="pbc")
    for occ in np.ndindex(dp):
        if np.allclose(occ, occ_idx):
            assert (np.allclose(MPX.element(mps, occ, bc="pbc"), 1.0))
        else:
            assert (np.allclose(MPX.element(mps, occ, bc="pbc"), 0.0))


def test_mul2():
    dp = [1, 5, 4, 2, 4]
    mps = dMPX.rand(dp, 7, bc="obc")
    alpha1 = np.random.random() - 0.5
    res = MPX.mul(alpha1, mps)
    mps_f = MPX.asfull(mps)
    res_from_mps_f = mps_f * alpha1
    res_f = MPX.asfull(res)
    assert(np.allclose(res_f, res_from_mps_f))
   
    dp = [1, 2]
    mps = dMPX.rand(dp, 2, bc="obc", dtype=np.complex)
    alpha1 = 1j
    print mps[0].dtype
    res = MPX.mul(alpha1, mps)
    mps_f = MPX.asfull(mps)
    res_from_mps_f = mps_f * alpha1
    res_f = MPX.asfull(res)

    dp = [1, 2]
    mps = dMPX.rand(dp, 2, bc="pbc", dtype=np.complex)
    alpha1 = 1j
    res = MPX.mul(alpha1, mps)
    mps_f = MPX.asfull(mps)
    res_from_mps_f = mps_f * alpha1
    res_f = MPX.asfull(res)
    assert(np.allclose(res_f, res_from_mps_f))

def test_element():
    dp = (4, 5, 3, 3, 3)
    mps = dMPX.product_state(dp, [0, 1, 2, 1, 1], bc="obc")
    vec = MPX.asfull(mps)

    for i, occ in enumerate(np.ndindex(dp)):
        assert abs(MPX.element(mps, occ) - vec[i]) < 1.e-10

    mps = dMPX.product_state(dp, [0, 1, 2, 2, 2], D=4, bc="pbc")
    vec = MPX.asfull(mps)

    for i, occ in enumerate(np.ndindex(dp)):
        assert abs(np.linalg.norm(MPX.element(mps, occ) - vec[i])) < 1.e-10

def test_mul():
    dps = [1, 5, 4]
    mps1 = dMPX.rand(dps, 4, bc="obc")

    alpha = -1
    mps2 = MPX.mul(alpha, mps1)
  
    norm1 = MPX.norm(mps1)
    norm2 = MPX.norm(mps2)
    ovlp  = MPX.dot(mps1.conj(),mps2)

    print 'scaling used: ', np.abs(alpha), np.angle(alpha)
    print 'ratio of norms: ', norm2 / norm1
    print 'scaling', ovlp / (norm1**2)
    assert(np.allclose(norm1, norm2))
    assert(np.allclose(ovlp / (norm1 ** 2), alpha))

def test_overwrite():
    dp = (1,3,2,2)
    np.random.seed(417)
    mps_obc = dMPX.rand(dp, D=7, bc="obc")
    mps_pbc = dMPX.zeros(dp, D=7, bc="pbc")
    dMPX.overwrite(mps_obc, out=mps_pbc)

    for occ in np.ndindex(dp):
        assert np.allclose(MPX.element(mps_obc, occ), MPX.element(mps_pbc, occ))

def test_norm():
    dp = (1,3,2,2)
    np.random.seed(417)
    mps_obc = dMPX.rand(dp, D=7, bc="obc")
    mps_pbc = dMPX.zeros(dp, D=7, bc="pbc")
    dMPX.overwrite(mps_obc, out=mps_pbc)
    
    norm_o = np.linalg.norm(MPX.asfull(mps_obc))
    norm_p = np.linalg.norm(MPX.asfull(mps_pbc))

    norm_o1 = MPX.norm(mps_obc)
    norm_p1 = MPX.norm(mps_pbc)
    assert np.allclose(norm_o, norm_o1)
    assert np.allclose(norm_p, norm_p1)

def test_add():
    dps = (2,2,2,2)
    #np.random.seed(417)
    mps_obc = dMPX.rand(dps, D=2, bc="obc")
    mps_obc2 = dMPX.rand(dps, D=2, bc="obc")
    mps_pbc = dMPX.rand(dps, D=2, bc="pbc")
    mps_pbc2 = dMPX.rand(dps, D=2, bc="pbc")
    
    assert np.allclose(np.linalg.norm(MPX.asfull(mps_obc)+MPX.asfull(mps_obc2)),
                       MPX.norm(dMPX.add(mps_obc,mps_obc2)))
    assert np.allclose(np.linalg.norm(MPX.asfull(mps_pbc)+MPX.asfull(mps_pbc2)),
                       MPX.norm(dMPX.add(mps_pbc,mps_pbc2)))
    assert np.allclose(np.linalg.norm(MPX.asfull(mps_obc)+MPX.asfull(mps_pbc2)),
                       MPX.norm(dMPX.add(mps_obc,mps_pbc2)))
        
def test_compress_obc():
    dps = [1,3,2,2]
    mps1 = dMPX.rand(dps, D=7, bc="obc")

    print [m.shape for m in mps1]
    
    mps_diff1 = dMPX.add(MPX.mul(-1,mps1), mps1)
    assert not(np.isnan(MPX.norm(mps1))), MPX.norm(mps1)
    assert MPX.norm(mps_diff1)<1.0e-12, MPX.norm(mps_diff1)

    # check dimension preserving
    mps11 = dMPX.add(mps1,mps1)
    mps11,dwt = MPX.compress(mps11,0,preserve_dim="true")
    mps11_,dwt = MPX.compress(mps11,0,preserve_dim="false")
    print [m.shape for m in mps11]
    print [m.shape for m in mps11_]
    print [m.shape for m in mps1]
    
    print MPX.norm(mps11)
    assert(dwt == 0), dwt
    
    mps_diff = dMPX.add(MPX.mul(-2,mps1),mps11)
    print abs(MPX.norm(mps_diff)/MPX.norm(mps11))<1.0e-7, MPX.norm(mps_diff)

    mps_diff = dMPX.add(MPX.mul(-2,mps1),mps11_)
    print abs(MPX.norm(mps_diff)/MPX.norm(mps11_))<1.0e-7, MPX.norm(mps_diff)

def test_compress_pbc():
    np.random.seed(417)
    
    dps = [1,3,2,2]
    mps_obc = dMPX.rand(dps, D=7, bc="obc")
    print "full dim", MPX.obc_dim(dps)
    mps1 = dMPX.zeros(dps, D=7, bc="pbc")
    dMPX.overwrite(mps_obc, out=mps1)
    print MPX.norm(mps1), MPX.norm(mps_obc)
    
    mps2 = dMPX.add(mps1,mps1)
    mps11_0,dwt = MPX.compress(mps2,0,preserve_dim="true")
    mps11_1,dwt = MPX.compress(mps2,1,preserve_dim="true")
    mps11_2,dwt = MPX.compress(mps2,2,preserve_dim="true")
    mps11_2_,dwt = MPX.compress(mps2,2,preserve_dim="false")
    mps11_4,dwt = MPX.compress(mps2,4,preserve_dim="true")
    mps11_8,dwt = MPX.compress(mps2,8,preserve_dim="false")

    mps_diff0 = dMPX.add(MPX.mul(-1,mps11_0),mps2)
    mps_diff1 = dMPX.add(MPX.mul(-1,mps11_1),mps2)
    mps_diff2 = dMPX.add(MPX.mul(-1,mps11_2),mps2)
    mps_diff2_ = dMPX.add(MPX.mul(-1,mps11_2_),mps2)
    mps_diff4 = dMPX.add(MPX.mul(-1,mps11_4),mps2)
    mps_diff8 = dMPX.add(MPX.mul(-1,mps11_8),mps2)

    mps4o,dwt = MPX.compress(dMPX.add(mps1,mps1), 4)
    mps_diff4o = dMPX.add(MPX.mul(-1,mps4o),mps2)
    
    print "D full", abs(MPX.norm(mps_diff0))
    print "D=1", abs(MPX.norm(mps_diff1))
    print "D=2", abs(MPX.norm(mps_diff2))
    print "D=2 (trunc)", abs(MPX.norm(mps_diff2_))
    print "D=4", abs(MPX.norm(mps_diff4))
    print "D=4 obc", abs(MPX.norm(mps_diff4o))
    print "D=8", abs(MPX.norm(mps_diff8))

    assert(abs(MPX.norm(mps_diff0))<1.0e-7)

def test_dot():
    dp = (4, 5, 3)

    # test mpo x mpo 
    mpo = dMPX.rand(zip(dp, dp), D=3, bc="obc")
    mat = MPX.asfull(mpo)
    
    mpo2 = MPX.dot(mpo, mpo)
    mat2 = np.dot(mat, mat)

    print np.linalg.norm(mat2)
    print np.linalg.norm(MPX.asfull(mpo2))
    
    assert np.linalg.norm(mat2 - MPX.asfull(mpo2)) < 1.e-9

    # test mpo x mps
    mps = dMPX.rand(dp, D=3, bc="pbc")
    vec = MPX.asfull(mps)
    matvec = np.dot(mat, vec)
    mps1 = MPX.dot(mpo, mps)

    assert np.linalg.norm(matvec - MPX.asfull(mps1)) < 1.e-9

    # test mps x mpo
    mps1 = MPX.dot(mps, mpo)
    vecmat = np.dot(vec, mat)
    assert np.linalg.norm(vecmat - MPX.asfull(mps1)) < 1.e-9
    
    # test mps x mps
    norm1 = MPX.dot(mps, mps)
    norm = np.dot(vec, vec)
    assert abs(norm - norm1) < 1.e-9

def test_dot2():
    dps = [1,4,4,2]
    mps1 = dMPX.rand(dps,4,bc="obc")
    mps2 = dMPX.rand(dps,3,bc="pbc")

    dps_o = [(d,d) for d in dps]
    mpo1 = dMPX.rand(dps_o,4,bc="pbc")
    mpo2 = dMPX.rand(dps_o,3,bc="obc")

    mpo3 = dMPX.rand(dps_o,3,bc="pbc")
    ## <mps1|mps2>
    ss11 = MPX.dot(mps1,mps1)
    ss12 = MPX.dot(mps1,mps2)
    ss22 = MPX.dot(mps2,mps2)
    assert np.allclose(ss11, np.dot(MPX.asfull(mps1), MPX.asfull(mps1)))
    assert np.allclose(ss12, np.dot(MPX.asfull(mps1), MPX.asfull(mps2)))
    assert np.allclose(ss22, np.dot(MPX.asfull(mps2), MPX.asfull(mps2)))

    ## mpo1|mps1>
    os11 = MPX.dot(mpo1,mps1)
    assert np.allclose(MPX.asfull(os11), np.dot(MPX.asfull(mpo1), MPX.asfull(mps1)))

    ## mpo2*mpo1
    oo21 = MPX.dot(mpo2,mpo1)
    assert np.allclose(MPX.asfull(oo21), np.dot(MPX.asfull(mpo2), MPX.asfull(mpo1)))

    ## <mps1|mpo1
    so11 = MPX.dot(mps1,mpo1)
    os32 = MPX.dot(mpo3,mps2)
    so23 = MPX.dot(mps2,mpo3)

    assert np.allclose(MPX.asfull(so11), np.dot(MPX.asfull(mps1), MPX.asfull(mpo1)))
    assert np.allclose(MPX.asfull(os32), np.dot(MPX.asfull(mpo3), MPX.asfull(mps2)))
    assert np.allclose(MPX.asfull(so23), np.dot(MPX.asfull(mps2), MPX.asfull(mpo3)))


##################
# these functions have not been fully tested 
def test_flatten():
    dp = [(1,2), (5,3), (4,4)]
    mpos = dMPX.rand(dp, 4)
    mps_out = MPX.flatten(mpos)
    for i in xrange(len(mps_out)):
        assert(mps_out[i].ndim == 3)
    

def test_dotcompress():
    dps = [1,2,3,4,2]
    dpo = [(d,d) for d in dps]
    mps = dMPX.rand(dps)
    mpo = dMPX.rand(dpo)
 
    mps2 = MPX.dot(mpo,mps)
    mpsc, errc = MPX.compress(mps2,4)

    mpsdc, errdc = MPX.dot_compress(mpo,mps,4)

    print errc, errdc, errdc-errc

    diff_mps = dMPX.add(MPX.mul(-1,mpsc),mpsdc)
    assert(MPX.norm(diff_mps) < 1.0e-8), MPX.norm(diff_mps)

    ### for mpo/mpo tests, need to flatten mpo before compression

def test_all():
    test_rand()
    test_obc_dim()
    test_product_state()
    test_mul2()
    test_element()
    test_mul()
    test_overwrite()
    test_norm()
    test_add()
    test_compress_obc()
    test_compress_pbc()
    test_dot()
    test_dot2()
