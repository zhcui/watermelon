import MPSblas

def test_create():
    dps = [1, 5, 4]
    mps = MPSblas.create(dps, 4)

    for x in mps:
        print x.shape


def test_calc_dim():
    dps = [5, 1, 4]
    drs = MPSblas.calc_dim(dps, None)
    print drs

    drs = MPSblas.calc_dim(dps, D=3)
    print drs

def test_product_state():
    dps = [4, 5, 3]
    try:
        mps = MPSblas.product_state(dps, [5, 5, 4])
    except:
        print "ok"

    mps = MPSblas.product_state(dps, [0, 1, 2])

    print MPSblas.ceval(mps, [0, 1, 2])
    print mps

def test_scal():
    dps = [1, 5, 4]
    mps1 = MPSblas.create(dps, 4)

    mps2 = MPSblas.scal(3., mps1)

    print MPSblas.norm(mps1)
    print MPSblas.norm(mps2)
    
