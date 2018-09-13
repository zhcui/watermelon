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
    mps = MPS.product_state(dp, [0, 1, 2, 2, 2], D=4, bc="pbc")
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

    print mps_pbc.__class__.__name__
    
    norm_o = np.linalg.norm(MPS.asfull(mps_obc))
    norm_p = np.linalg.norm(MPS.asfull(mps_pbc))

    norm_o1 = MPS.norm(mps_obc)
    norm_p1 = MPS.norm(mps_pbc)
    assert np.allclose(norm_o, norm_o1)
    assert np.allclose(norm_p, norm_p1)
