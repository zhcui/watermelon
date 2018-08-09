import numpy as np
import MPSblas

def test_rand():
    dp = [1, 5, 4]
    mps = MPSblas.rand(dp, 4)

    dp = [(1,2), (5,3), (4,4)]
    mpo = MPSblas.rand(dp, 4)
    for x in mps:
        print x.shape

    for x in mpo:
        print x.shape


def test_calc_dim():
    dps = [5, 1, 4]
    drs = MPSblas.calc_dim(dps, None)
    print drs

    drs = MPSblas.calc_dim(dps, D=3)
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
        print occ, MPSblas.element(mps, occ)
    print mps

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

    
    
