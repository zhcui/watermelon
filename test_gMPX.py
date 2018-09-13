#! /usr/bin/env python
import numpy as np
import pbc_MPS as MPS
from pbc_MPS import asfull
import pytest

def test_rand():
    dp = [1, 5, 4, 2, 4]
    mps = MPS.rand(dp, 7, bc="obc")
    mps_res_shape = [(1, 1, 1),
                     (1, 5, 5),
                     (5, 4, 7),
                     (7, 2, 4),
                     (4, 4, 1)]
    for i, x in enumerate(mps):
        assert(np.allclose(x.shape, mps_res_shape[i]))

    dp = [1, 5, 4, 2, 4]
    mps = MPS.rand(dp, 7, bc="pbc")
    mps_res_shape = [(7, 1, 7),
                     (7, 5, 7),
                     (7, 4, 7),
                     (7, 2, 7),
                     (7, 4, 7)]
    
    for i, x in enumerate(mps):
        assert(np.allclose(x.shape, mps_res_shape[i]))
    
    print "mps"
    
    dp = [(1,2), (5,3), (4,4)]
    mpo = MPS.rand(dp, 4, bc="obc")
    mpo_res_shape = [(1, 1, 2, 2),
                     (2, 5, 3, 4),
                     (4, 4, 4, 1)]
    
    print "mpo"
    for i, x in enumerate(mpo):
        print x.shape
        assert(np.allclose(x.shape, mpo_res_shape[i]))

def test_obc_dim():
    dps = [5, 1, 4]
    drs = MPS.obc_dim(dps, None)
    print drs
    assert(np.allclose(drs, [4, 4]))

    drs = MPS.obc_dim(dps, D = 3)
    print drs
    assert(np.allclose(drs, [3, 3]))


def test_product_state():
    dp = (4, 5, 3)

    # out of range case
    occ_idx = (5, 5, 4)
    with pytest.raises(IndexError):
        mps = MPS.product_state(dp, occ_idx, D=1, bc="obc")
    with pytest.raises(IndexError):
        mps = MPS.product_state(dp, occ_idx, D=1, bc="pbc")

    occ_idx = (0, 1, 2)
    mps = MPS.product_state(dp, occ_idx, D=1, bc="obc")
    for occ in np.ndindex(dp):
        if np.allclose(occ, occ_idx):
            assert (np.allclose(MPS.element(mps, occ, bc="obc"), 1.0))
        else:
            assert (np.allclose(MPS.element(mps, occ, bc="obc"), 0.0))

    occ_idx = (0, 1, 2)
    mps = MPS.product_state(dp, occ_idx, D=4, bc="obc")
    for occ in np.ndindex(dp):
        if np.allclose(occ, occ_idx):
            assert (np.allclose(MPS.element(mps, occ, bc="obc"), 1.0))
        else:
            assert (np.allclose(MPS.element(mps, occ, bc="obc"), 0.0))

    occ_idx = (0, 1, 2)
    mps = MPS.product_state(dp, occ_idx, D=4, bc="pbc")
    for occ in np.ndindex(dp):
        if np.allclose(occ, occ_idx):
            assert (np.allclose(MPS.element(mps, occ, bc="pbc"), 1.0))
        else:
            assert (np.allclose(MPS.element(mps, occ, bc="pbc"), 0.0))


def test_mul2():
    dp = [1, 5, 4, 2, 4]
    mps = MPS.rand(dp, 7, bc="obc")
    alpha1 = np.random.random() - 0.5
    res = MPS.mul(alpha1, mps)
    mps_f = MPS.asfull(mps)
    res_from_mps_f = mps_f * alpha1
    res_f = MPS.asfull(res)
    assert(np.allclose(res_f, res_from_mps_f))
   
    dp = [1, 2]
    mps = MPS.rand(dp, 2, bc="obc")
    alpha1 = 1j
    res = MPS.mul(alpha1, mps)
    mps_f = MPS.asfull(mps)
    res_from_mps_f = mps_f * alpha1
    res_f = MPS.asfull(res)

    dp = [1, 2]
    mps = MPS.rand(dp, 2, bc="pbc")
    alpha1 = 1j
    res = MPS.mul(alpha1, mps)
    mps_f = MPS.asfull(mps)
    res_from_mps_f = mps_f * alpha1
    res_f = MPS.asfull(res)
    assert(np.allclose(res_f, res_from_mps_f))

def test_element():
    dp = (4, 5, 3, 3, 3)
    mps = MPS.product_state(dp, [0, 1, 2, 1, 1], bc="obc")
    vec = MPS.asfull(mps)

    for i, occ in enumerate(np.ndindex(dp)):
        assert abs(MPS.element(mps, occ) - vec[i]) < 1.e-10

    mps = MPS.product_state(dp, [0, 1, 2, 2, 2], D=4, bc="pbc")
    vec = MPS.asfull(mps)

    for i, occ in enumerate(np.ndindex(dp)):
        assert abs(np.linalg.norm(MPS.element(mps, occ) - vec[i])) < 1.e-10

def test_mul():
    dps = [1, 5, 4]
    mps1 = MPS.rand(dps, 4, bc="obc")

    alpha = -1
    mps2 = MPS.mul(alpha, mps1)
  
    norm1 = MPS.norm(mps1)
    norm2 = MPS.norm(mps2)
    ovlp  = MPS.dot(mps1.conj(),mps2)

    print 'scaling used: ', np.abs(alpha), np.angle(alpha)
    print 'ratio of norms: ', norm2 / norm1
    print 'scaling', ovlp / (norm1**2)
    assert(np.allclose(norm1, norm2))
    assert(np.allclose(ovlp / (norm1 ** 2), alpha))

def test_overwrite():
    dp = (1,3,2,2)
    np.random.seed(417)
    mps_obc = MPS.rand(dp, D=7, bc="obc")
    mps_pbc = MPS.zeros(dp, D=7, bc="pbc")
    MPS.overwrite(mps_obc, out=mps_pbc)

    for occ in np.ndindex(dp):
        assert np.allclose(MPS.element(mps_obc, occ), MPS.element(mps_pbc, occ))

def test_norm():
    dp = (1,3,2,2)
    np.random.seed(417)
    mps_obc = MPS.rand(dp, D=7, bc="obc")
    mps_pbc = MPS.zeros(dp, D=7, bc="pbc")
    MPS.overwrite(mps_obc, out=mps_pbc)
    
    norm_o = np.linalg.norm(MPS.asfull(mps_obc))
    norm_p = np.linalg.norm(MPS.asfull(mps_pbc))

    norm_o1 = MPS.norm(mps_obc)
    norm_p1 = MPS.norm(mps_pbc)
    assert np.allclose(norm_o, norm_o1)
    assert np.allclose(norm_p, norm_p1)

def test_add():
    dps = (2,2,2,2)
    #np.random.seed(417)
    mps_obc = MPS.rand(dps, D=2, bc="obc")
    mps_obc2 = MPS.rand(dps, D=2, bc="obc")
    mps_pbc = MPS.rand(dps, D=2, bc="pbc")
    mps_pbc2 = MPS.rand(dps, D=2, bc="pbc")
    
    assert np.allclose(np.linalg.norm(MPS.asfull(mps_obc)+MPS.asfull(mps_obc2)),
                       MPS.norm(MPS.add(mps_obc,mps_obc2)))
    assert np.allclose(np.linalg.norm(MPS.asfull(mps_pbc)+MPS.asfull(mps_pbc2)),
                       MPS.norm(MPS.add(mps_pbc,mps_pbc2)))
    assert np.allclose(np.linalg.norm(MPS.asfull(mps_obc)+MPS.asfull(mps_pbc2)),
                       MPS.norm(MPS.add(mps_obc,mps_pbc2)))
        
def test_compress_obc():
    dps = [1,3,2,2]
    mps1 = MPS.rand(dps, D=7, bc="obc")

    print [m.shape for m in mps1]
    
    mps_diff1 = MPS.add(MPS.mul(-1,mps1), mps1)
    assert not(np.isnan(MPS.norm(mps1))), MPS.norm(mps1)
    assert MPS.norm(mps_diff1)<1.0e-12, MPS.norm(mps_diff1)

    # check dimension preserving
    mps11 = MPS.add(mps1,mps1)
    mps11,dwt = MPS.compress(mps11,0,preserve_dim="true")
    mps11_,dwt = MPS.compress(mps11,0,preserve_dim="false")
    print [m.shape for m in mps11]
    print [m.shape for m in mps11_]
    print [m.shape for m in mps1]
    
    print MPS.norm(mps11)
    assert(dwt == 0), dwt
    
    mps_diff = MPS.add(MPS.mul(-2,mps1),mps11)
    print abs(MPS.norm(mps_diff)/MPS.norm(mps11))<1.0e-7, MPS.norm(mps_diff)

    mps_diff = MPS.add(MPS.mul(-2,mps1),mps11_)
    print abs(MPS.norm(mps_diff)/MPS.norm(mps11_))<1.0e-7, MPS.norm(mps_diff)

def test_compress_pbc():
    np.random.seed(417)
    
    dps = [1,3,2,2]
    mps_obc = MPS.rand(dps, D=7, bc="obc")
    print "full dim", MPS.obc_dim(dps)
    mps1 = MPS.zeros(dps, D=7, bc="pbc")
    MPS.overwrite(mps_obc, out=mps1)
    print MPS.norm(mps1), MPS.norm(mps_obc)
    
    mps2 = MPS.add(mps1,mps1)
    mps11_0,dwt = MPS.compress(mps2,0,preserve_dim="true")
    mps11_1,dwt = MPS.compress(mps2,1,preserve_dim="true")
    mps11_2,dwt = MPS.compress(mps2,2,preserve_dim="true")
    mps11_2_,dwt = MPS.compress(mps2,2,preserve_dim="false")
    mps11_4,dwt = MPS.compress(mps2,4,preserve_dim="true")
    mps11_8,dwt = MPS.compress(mps2,8,preserve_dim="false")

    mps_diff0 = MPS.add(MPS.mul(-1,mps11_0),mps2)
    mps_diff1 = MPS.add(MPS.mul(-1,mps11_1),mps2)
    mps_diff2 = MPS.add(MPS.mul(-1,mps11_2),mps2)
    mps_diff2_ = MPS.add(MPS.mul(-1,mps11_2_),mps2)
    mps_diff4 = MPS.add(MPS.mul(-1,mps11_4),mps2)
    mps_diff8 = MPS.add(MPS.mul(-1,mps11_8),mps2)

    mps4o,dwt = MPS.compress(MPS.add(mps1,mps1), 4)
    mps_diff4o = MPS.add(MPS.mul(-1,mps4o),mps2)
    
    print "D full", abs(MPS.norm(mps_diff0))
    print "D=1", abs(MPS.norm(mps_diff1))
    print "D=2", abs(MPS.norm(mps_diff2))
    print "D=2 (trunc)", abs(MPS.norm(mps_diff2_))
    print "D=4", abs(MPS.norm(mps_diff4))
    print "D=4 obc", abs(MPS.norm(mps_diff4o))
    print "D=8", abs(MPS.norm(mps_diff8))

    assert(abs(MPS.norm(mps_diff0))<1.0e-7)

def test_dot():
    dp = (4, 5, 3)

    # test mpo x mpo 
    mpo = MPS.rand(zip(dp, dp))
    mat = MPS.asfull(mpo)
    
    mpo2 = MPS.dot(mpo, mpo)
    mat2 = np.dot(mat, mat)

    print np.linalg.norm(mat2)
    print np.linalg.norm(MPS.asfull(mpo2))
    
    assert np.linalg.norm(mat2 - MPS.asfull(mpo2)) < 1.e-9

    # test mpo x mps
    mps = MPS.rand(dp)
    vec = MPS.asfull(mps)
    matvec = np.dot(mat, vec)
    mps1 = MPS.dot(mpo, mps)

    assert np.linalg.norm(matvec - MPS.asfull(mps1)) < 1.e-9

    # test mps x mpo
    mps1 = MPS.dot(mps, mpo)
    vecmat = np.dot(vec, mat)
    assert np.linalg.norm(vecmat - MPS.asfull(mps1)) < 1.e-9
    
    # test mps x mps
    norm1 = MPS.dot(mps, mps)
    norm = np.dot(vec, vec)
    assert abs(norm - norm1) < 1.e-9

def test_dot():
    dps = [1,4,4,2]
    mps1 = MPS.rand(dps,4,bc="obc")
    mps2 = MPS.rand(dps,3,bc="pbc")

    dps_o = [(d,d) for d in dps]
    mpo1 = MPS.rand(dps_o,4,bc="pbc")
    mpo2 = MPS.rand(dps_o,3,bc="obc")

    mpo3 = MPS.rand(dps_o,3,bc="pbc")
    ## <mps1|mps2>
    ss11 = MPS.dot(mps1,mps1)
    ss12 = MPS.dot(mps1,mps2)
    ss22 = MPS.dot(mps2,mps2)
    assert np.allclose(ss11, np.dot(MPS.asfull(mps1), MPS.asfull(mps1)))
    assert np.allclose(ss12, np.dot(MPS.asfull(mps1), MPS.asfull(mps2)))
    assert np.allclose(ss22, np.dot(MPS.asfull(mps2), MPS.asfull(mps2)))

    ## mpo1|mps1>
    os11 = MPS.dot(mpo1,mps1)
    assert np.allclose(MPS.asfull(os11), np.dot(MPS.asfull(mpo1), MPS.asfull(mps1)))

    ## mpo2*mpo1
    oo21 = MPS.dot(mpo2,mpo1)
    assert np.allclose(MPS.asfull(oo21), np.dot(MPS.asfull(mpo2), MPS.asfull(mpo1)))

    ## <mps1|mpo1
    so11 = MPS.dot(mps1,mpo1)
    os32 = MPS.dot(mpo3,mps2)
    so23 = MPS.dot(mps2,mpo3)

    assert np.allclose(MPS.asfull(so11), np.dot(MPS.asfull(mps1), MPS.asfull(mpo1)))
    assert np.allclose(MPS.asfull(os32), np.dot(MPS.asfull(mpo3), MPS.asfull(mps2)))
    assert np.allclose(MPS.asfull(so23), np.dot(MPS.asfull(mps2), MPS.asfull(mpo3)))


##################
# these functions have not been fully tested 
def test_flatten():
    dp = [(1,2), (5,3), (4,4)]
    mpos = MPS.rand(dp, 4)
    mps_out = MPS.flatten(mpos)
    for i in xrange(len(mps_out)):
        assert(mps_out[i].ndim == 3)
    

def test_dotcompress():
    dps = [1,2,3,4,2]
    dpo = [(d,d) for d in dps]
    mps = MPS.rand(dps)
    mpo = MPS.rand(dpo)
 
    mps2 = MPS.dot(mpo,mps)
    mpsc, errc = MPS.compress(mps2,4)

    mpsdc, errdc = MPS.dot_compress(mpo,mps,4)

    print errc, errdc, errdc-errc

    diff_mps = MPS.add(MPS.mul(-1,mpsc),mpsdc)
    assert(MPS.norm(diff_mps) < 1.0e-8), MPS.norm(diff_mps)

    ### for mpo/mpo tests, need to flatten mpo before compression




    
