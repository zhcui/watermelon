import numpy as np
import gMPX as MPX
import sMPX
import dMPX
def test_rand():
    dp = [1, 5, 4, 2, 4]
    mps = sMPX.rand(dp, 7, bc="obc")
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
    mps1 = dMPX.product_state(dp, occ_idx, D=4, bc="obc")
    mps2 = dMPX.add(mps1, mps1)
    mps3 = dMPX.add(mps2, mps1)
    smps1 = sMPX.from_dense(mps1)
    smps3 = sMPX.from_dense(mps3)

    for i, m in enumerate(smps1):
        print i, m.coords.shape, m.shape, len(m.data), m.nnz
        assert m.nnz == 1
    for i, m in enumerate(smps3):
        print i, m.coords.shape, m.shape, len(m.data), m.nnz
        assert m.nnz == 3

def test_element():
    dp = (2, 2, 3, 3, 3)
    mps = dMPX.product_state(dp, [0, 1, 2, 2, 2], D=2, bc="obc")
    smps = sMPX.from_dense(mps)
    vec = MPX.asfull(mps)

    assert(MPX.element(smps, [0, 1, 2, 2, 2]) ==
           MPX.element(mps, [0, 1, 2,2,2]))

    for i, occ in enumerate(np.ndindex(dp)):
        print occ, MPX.element(smps, occ), vec[i]
        assert abs(np.linalg.norm(MPX.element(smps, occ) - vec[i])) < 1.e-10

def test_asfull():
    dp = [1, 5, 4, 2, 4]
    smps = sMPX.rand(dp, 7, bc="pbc")
    mps = sMPX.todense(smps)
    print np.linalg.norm(MPX.asfull(mps)), np.linalg.norm(MPX.asfull(smps))
    assert np.allclose(np.linalg.norm(MPX.asfull(mps)-MPX.asfull(smps)), 0.)

    mps = sMPX.rand(dp, 7, bc="obc")
    smps = sMPX.from_dense(mps)
    assert np.allclose(np.linalg.norm(MPX.asfull(mps)-MPX.asfull(smps)), 0.)

def test_mul2():
    dp = [1, 5, 4, 2, 4]
    mps = sMPX.rand(dp, 7, bc="obc")
    alpha1 = np.random.random() - 0.5
    res = MPX.mul(alpha1, mps)
    
    assert((mps[0] * alpha1).__class__.__name__=="COO")
    print res[1]
    mps_f = MPX.asfull(mps)
    res_from_mps_f = mps_f * alpha1
    res_f = MPX.asfull(res)
    assert(np.allclose(res_f, res_from_mps_f))

    dp = [1, 2]
    mps = sMPX.rand(dp, 2, bc="obc")
    alpha1 = 1j
    res = MPX.mul(alpha1, mps)
    mps_f = MPX.asfull(mps)
    res_from_mps_f = mps_f * alpha1
    res_f = MPX.asfull(res)

    dp = [1, 2]
    mps = sMPX.rand(dp, 2, bc="pbc")
    alpha1 = 1j + 3
    res = MPX.mul(alpha1, mps)
    print res[0]
    
    mps_f = MPX.asfull(mps)
    res_from_mps_f = mps_f * alpha1
    res_f = MPX.asfull(res)
    assert(np.allclose(res_f, res_from_mps_f))

def test_norm():
    dp = (2,3,2)
    np.random.seed(417)
    mps_obc = sMPX.rand(dp, D=3, bc="obc")
    print mps_obc[0]
    mps_full = MPX.asfull(mps_obc)
    print np.linalg.norm(mps_full)
    print "norm", MPX.norm(mps_obc)
    
    mps_pbc = sMPX.zeros(dp, D=7, bc="pbc")
    sMPX.overwrite(mps_obc, out=mps_pbc)

    norm_o = np.linalg.norm(MPX.asfull(mps_obc))
    norm_p = np.linalg.norm(MPX.asfull(mps_pbc))

    norm_o1 = MPX.norm(mps_obc)
    norm_p1 = MPX.norm(mps_pbc)
    print norm_o, norm_o1, norm_p, norm_p1
    assert np.allclose(norm_o, norm_o1)
    assert np.allclose(norm_p, norm_p1)

def test_add():
    dps = (2,2,2,2)
    #np.random.seed(417)
    mps_obc = sMPX.rand(dps, D=4, bc="obc")
    mps_obc2 = sMPX.rand(dps, D=4, bc="obc")
    mps_pbc = sMPX.rand(dps, D=4, bc="pbc")
    mps_pbc2 = sMPX.rand(dps, D=4, bc="pbc")

    print mps_obc[0]
    print mps_obc[0].shape
    print mps_obc[0].coords
    print mps_obc[0].data

    assert np.allclose(np.linalg.norm(MPX.asfull(mps_obc)+MPX.asfull(mps_obc2)),
                       MPX.norm(sMPX.add(mps_obc,mps_obc2)))
    assert np.allclose(np.linalg.norm(MPX.asfull(mps_pbc)+MPX.asfull(mps_pbc2)),
                       MPX.norm(sMPX.add(mps_pbc,mps_pbc2)))
    assert np.allclose(np.linalg.norm(MPX.asfull(mps_obc)+MPX.asfull(mps_pbc2)),
                       MPX.norm(sMPX.add(mps_obc,mps_pbc2)))

def test_compress():
    #np.random.seed(417)
    dps = [2]*10
    mps0 = dMPX.rand(dps, D=7, bc="obc")

    smps1 = sMPX.from_dense(mps0)
    print "Check sparse", [m.__class__.__name__ for m in smps1]
    # check dimension preserving
    mps00 = dMPX.add(mps0,mps0)
    smps11 = sMPX.add(smps1,smps1)

    print MPX.norm(mps00), MPX.norm(smps11)

    print "Initial dimensions", [m.shape for m in mps00]
    # compress
    for D in (2,3,7):
        mps00c = MPX.compress(mps00,D,preserve_dim="false")
        print "After DENSE compress", D, MPX.norm(mps00c)
        #print "Compressed dimensions", [m.shape for m in mps00c]

    print "IN SPARSE part"
    for D in (2,3,7):
        smps11c = MPX.compress(smps11, D,preserve_dim="false")
        print "After SPARSE compress", D, MPX.norm(smps11c)
        #print "Check sparse", [m.__class__.__name__ for m in smps11c]
        #print "Compressed dimensions", [m.shape for m in smps11c]
    

if __name__=="__main__":
    test_all()
    
def test_all():
    # test_rand()
    # print "1"
    # test_product_state()
    # print "2"
    # test_element()
    # print "3"
    # test_asfull()
    # print "4"
    # test_mul2()
    # print "5"
    # test_norm()
    # print "6"
    # test_add()
    print "7"
    test_compress()
    print "8"
