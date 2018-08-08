import numpy as np

### create MPS object as ndarray of ndarrays
def create(dps,D=None):
    # dps is a list/array of integers specifying the dimension of the physical bonds at each site
    # cap MPS to virtual dimension D
    # returns MPS object with 

    L = len(dps)
    mps = np.empty(L,dtype=np.object)
    
    # calculate right bond dims of each tensor
    dim_rs = calc_dim(dps,D)

    # fill in MPS with random arrays of the correct shape
    mps[0]  = np.random.rand(1,dps[0],dim_rs[0])
    for i in range(1,L-1):
        mps[i] = np.random.rand(dim_rs[i-1],dps[i],dim_rs[i])
    mps[-1] = np.random.rand(dim_rs[-1],dps[-1],1)

    return mps


def calc_dim(dps,D):
    # dps is a list/array of integers specifying the dimension of the physical bonds at each site
    # cap MPS to virtual dimension D
    # returns a list of the the right bond dimensions to be used in generating the MPS

    dimR = np.cumprod(dps)
    dimL = np.cumprod(dps[::-1])[::-1]
    
    dimMin = np.minimum(dimR[:-1],dimL[1:])
    if D is not None:
        dimMin = np.minimum(dimMin,[D]*(len(dps)-1))

    return dimMin

def product_state(dps,occ):
    # dps:  list/array of integers specifying dimension of physical bonds at each site
    # occ:  occupancy vector (len L), numbers specifying physical bond index occupied
    # returns product state mps according to occ

    L = len(dps)
    mps = create(dps,1)

    for i in range(L):
        site = np.zeros((1,dps[i],1))
        site[0,occ[i],0] = 1.0
        mps[i] = site.copy()
    
    return mps

def scal(mps,alpha):
    # result:  mps scaled by alpha

    L = len(mps)
    const = float(abs(alpha))**(1./L)
    for i in range(L):
        mps[i] *= const
 
    # change sign as specified by alpha
    mps[0] *= np.sign(alpha)



def conj(mps):
    # result: takes complex conjugate of mps
    for i in range(len(mps)):
        mps[i] = np.conj(mps[i])


def axpy(alpha,mps1,mps2):
    # alpha = scalar, mps1,mps2 are ndarrays of tensors   
    # returns alpha*mps1 + mps2

    mps_new = np.empty(len(mps1),dtype=np.object)

    scal(mps1,alpha)
    assert(len(mps1)==len(mps2)), 'need to have same lengths: (%d,%d)'%(len(mps1),len(mps2))
    for i in range(len(mps1)):
        l1,n1,r1 = mps1[i].shape
        l2,n2,r2 = mps2[i].shape
        assert(n1==n2)
        newSite = np.zeros((l1+l2,n1,r1+r2))
        combine = (l1==l2,r1==r2)

        iL1,iR1 = l1,r1
        iL2,iR2 = l1,r1
 
        if combine[0]:           iL2 = 0
        if combine[1]:           iR2 = 0

        newSite[:iL1,:,:iR1] = mps1[i]
        newSite[iL2:,:,iR2:] = mps2[i]

        mps_new[i] = newSite.copy()
    
    return mps_new
    

def compress(mps,D):
    pass
    
def inprod(mps1,mpo,mps2):
    pass


