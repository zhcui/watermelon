import numpy as np

### create MPS object as ndarray of ndarrays
def create(dps,D):
    # dps is a list/array of integers specifying the dimension of the physical bonds at each site
    # cap MPS to virtual dimension D
    # returns MPS object with 

    L = len(dps)
    mps = np.array(L,dtype=np.object)
    
    # calculate right bond dims of each tensor
    dim_rs = calcdim(dps,D)

    # fill in MPS with random arrays of the correct shape
    mps[0]  = np.random.rand(1,dps[0],dim_rs[0])
    for i in range(1,L-1):
        mps[i] = np.random.rand(dim_rs[i-1],dps[i],dim_rs[i])
    mps[-1] = np.random.rand(dps[1],1,dim_rs[-1])

    return mps


def calc_dim(dps,D):
    # dps is a list/array of integers specifying the dimension of the physical bonds at each site
    # cap MPS to virtual dimension D
    # returns a list of the the right bond dimensions to be used in generating the MPS

    dimR = np.cumprod(dps)
    dimL = np.cumprod(dps[::-1])[::-1]
    
    dimMin = np.minimum(dimR[:-1],dimL[1:])
    dimMin = np.minimum(dimMin,[D]*len(dps))

    return dimMin

def product_state(dps,occ)
    # dps:  list/array of integers specifying dimension of physical bonds at each site
    # occ:  occupancy vector (len L), numbers specifying physical bond index occupied
    # returns product state mps according to occ

    L = len(dps)
    mps = create(dps,1)

    for i in range(L):
        site = np.zeros((1,dps[i],1))
        site[0,occ[i],0) = 1.0
        mps[i] = site.copy()
    
    return mps

def scal(mps,alpha):
    # mps:   mps object to be scaled
    # alpha: scaling factor
    # returns mps scaled by alpha

    L = len(mps)
    const = float(alpha)**(1./L)
    for m in mps:
        m = m*const

    return mps


