import numpy as np
import pbc_MPS as MPS
import sp_MPS as sMPS

def test_rand():
    dp = [1, 5, 4, 2, 4]
    mps = sMPS.rand(dp, 7, bc="obc")
    mps_res_shape = [(1, 1, 1),
                     (1, 5, 5),
                     (5, 4, 7),
                     (7, 2, 4),
                     (4, 4, 1)]
    for i, x in enumerate(mps):
        assert(np.allclose(x.shape, mps_res_shape[i]))

def test_product_state():
    dp = (4, 5, 3, 2)
    occ_idx = (0, 1, 2, 1)
    mps1 = MPS.product_state(dp, occ_idx, D=4, bc="obc")
    mps2 = MPS.add(mps1, mps1)
    mps3 = MPS.add(mps2, mps1)
    smps1 = sMPS.from_dense(mps1)
    smps3 = sMPS.from_dense(mps3)

    for i, m in enumerate(smps1):
        print i, m.coords.shape, m.shape, len(m.data), m.nnz
        assert m.nnz == 1
    for i, m in enumerate(smps3):
        print i, m.coords.shape, m.shape, len(m.data), m.nnz
        assert m.nnz == 3

def test_element():
    dp = (4, 5, 3, 3, 3)
    mps = MPS.product_state(dp, [0, 1, 2, 2, 2], D=2, bc="obc")
    smps = sMPS.from_dense(mps)
    vec = MPS.asfull(smps)

    assert(MPS.element(smps, [0, 1, 2, 2, 2]) ==
           MPS.element(mps, [0, 1, 2,2,2]))

    for i, occ in enumerate(np.ndindex(dp)):
        assert abs(np.linalg.norm(MPS.element(smps, occ) - vec[i])) < 1.e-10

def test_asfull():
    dp = [1, 5, 4, 2, 4]
    smps = sMPS.rand(dp, 7, bc="pbc")
    mps = sMPS.todense(smps)
    assert np.allclose(np.linalg.norm(MPS.asfull(mps)-MPS.asfull(smps)), 0.)

    mps = MPS.rand(dp, 7, bc="obc")
    smps = sMPS.from_dense(mps)
    assert np.allclose(np.linalg.norm(MPS.asfull(mps)-MPS.asfull(smps)), 0.)

def test_mul2():
    dp = [1, 5, 4, 2, 4]
    mps = sMPS.rand(dp, 7, bc="obc")
    alpha1 = np.random.random() - 0.5
    res = MPS.mul(alpha1, mps)
    
    assert((mps[0] * alpha1).__class__.__name__=="COO")
    print res[1]
    mps_f = MPS.asfull(mps)
    res_from_mps_f = mps_f * alpha1
    res_f = MPS.asfull(res)
    assert(np.allclose(res_f, res_from_mps_f))

    dp = [1, 2]
    mps = sMPS.rand(dp, 2, bc="obc")
    alpha1 = 1j
    res = MPS.mul(alpha1, mps)
    mps_f = MPS.asfull(mps)
    res_from_mps_f = mps_f * alpha1
    res_f = MPS.asfull(res)

    dp = [1, 2]
    mps = sMPS.rand(dp, 2, bc="pbc")
    alpha1 = 1j + 3
    res = MPS.mul(alpha1, mps)
    print res[0]
    
    mps_f = MPS.asfull(mps)
    res_from_mps_f = mps_f * alpha1
    res_f = MPS.asfull(res)
    assert(np.allclose(res_f, res_from_mps_f))

def test_norm():
    dp = (1,3,2,2)
    np.random.seed(417)
    mps_obc = sMPS.rand(dp, D=7, bc="obc")
    mps_pbc = sMPS.zeros(dp, D=7, bc="pbc")
    sMPS.overwrite(mps_obc, out=mps_pbc)

    norm_o = np.linalg.norm(MPS.asfull(mps_obc))
    norm_p = np.linalg.norm(MPS.asfull(mps_pbc))

    norm_o1 = MPS.norm(mps_obc)
    norm_p1 = MPS.norm(mps_pbc)
    assert np.allclose(norm_o, norm_o1)
    assert np.allclose(norm_p, norm_p1)

def test_add():
    dps = (2,2,2,2)
    #np.random.seed(417)
    mps_obc = sMPS.rand(dps, D=4, bc="obc")
    mps_obc2 = sMPS.rand(dps, D=4, bc="obc")
    mps_pbc = sMPS.rand(dps, D=4, bc="pbc")
    mps_pbc2 = sMPS.rand(dps, D=4, bc="pbc")

    print mps_obc[0]
    print mps_obc[0].shape
    print mps_obc[0].coords
    print mps_obc[0].data

    assert np.allclose(np.linalg.norm(MPS.asfull(mps_obc)+MPS.asfull(mps_obc2)),
                       MPS.norm(sMPS.add(mps_obc,mps_obc2)))
    assert np.allclose(np.linalg.norm(MPS.asfull(mps_pbc)+MPS.asfull(mps_pbc2)),
                       MPS.norm(sMPS.add(mps_pbc,mps_pbc2)))
    assert np.allclose(np.linalg.norm(MPS.asfull(mps_obc)+MPS.asfull(mps_pbc2)),
                       MPS.norm(sMPS.add(mps_obc,mps_pbc2)))

def test_compress():
    np.random.seed(417)
    dps = [1,3,2,2]
    mps1 = sMPS.rand(dps, D=7, bc="obc")

    print [m.shape for m in mps1]

    # check dimension preserving
    mps11 = sMPS.add(mps1,mps1)
    print "data", mps11.data
    print "before compression", mps11[0].__class__.__name__
    print "MPS norm"
    print MPS.norm(mps1)
    print MPS.norm(mps11)

    
    mps11,dwt = MPS.compress(mps11,0,preserve_dim="true")
    mps11_,dwt = MPS.compress(mps11,0,preserve_dim="false")
    print [m.shape for m in mps11]
    print [m.shape for m in mps11_]
    print [m.shape for m in mps1]

    print "after compression", mps11[0].__class__.__name__
    print MPS.norm(mps11)
    assert(dwt == 0), dwt

    
    # mps1dense = sMPS.todense(mps1)
    # mps_diff = MPS.add(MPS.mul(-2,mps1dense),mps11)
    # print "---"
    # print abs(MPS.norm(mps_diff)/MPS.norm(mps11))<1.0e-7
    # print MPS.norm(mps_diff)
    # print "88"
    #mps_diff = sMPS.add(MPS.mul(-2,mps1),mps11_)
    #print abs(MPS.norm(mps_diff)/MPS.norm(mps11_))<1.0e-7, MPS.norm(mps_diff)

if __name__=="__main__":
    test_add()
