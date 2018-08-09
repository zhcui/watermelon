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

 
    
