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


def scal(alpha,mps):
    # result:  mps scaled by alpha

    L = len(mps)
    new_mps = np.empty(L,dtype=np.object)

    const = float(abs(alpha))**(1./L)
    for i in range(L):
        new_mps[i] = mps[i].copy()*const
 
    # change sign as specified by alpha
    new_mps[0] *= np.sign(alpha)

    return new_mps


def conj(mps):
    # result: takes complex conjugate of mps
    L = len(mps)
    new_mps = np.empty(L,dtype=np.object)

    for i in range(L):
        new_mps[i] = np.conj(mps[i])

    return new_mps


def axpy(alpha,mpx1,mpx2):
    # alpha = scalar, mps1,mps2 are ndarrays of tensors   
    # returns alpha*mps1 + mps2

    mps_new = np.empty(len(mps1),dtype=np.object)

    scal(mps1,alpha)
    assert(len(mpx1)==len(mpx2)), 'need to have same lengths: (%d,%d)'%(len(mpx1),len(mpx2))
    for i in range(len(mps1)):
        sh1 = mpx1[i].shape
        sh2 = mpx2[i].shape
        assert(n1[1:-1]==n2[1:-2]), 'need physical bonds at site %d to match'%(i)        

        l1,n1,r1 = sh1[0],np.prod(sh1[1:-1]),sh1[-1]
        l2,n2,r2 = sh2[0],np.prod(sh2[1:-1]),sh2[-1]

        newSite = np.zeros((l1+l2,n1,r1+r2))
        newSite[:l1,:,:r1] = mpx1[i].reshape(l1,n1,r1)
        newSite[l1:,:,r1:] = mpx2[i].reshape(l1,n2,r2)

        newSite = newSite.reshape(l1+l2,sh1[1:-1],r1+r2)

        if i==0:  newSite = np.einsum('l...r->...r',newSite).reshape((1,)+sh1[1:-1]+(r1+r2,))
        if i==L:  newSite = np.einsum('l...r->l...',newSite).reshape((l1+l2,)+sh1[1:-1]+(1,))

        mps_new[i] = newSite.copy()
    
    return mps_new
    

def inprod(arrow,mps1,mpo,mps2):\
    # returns <mps2|mpo|mps2>

    if   arrow == 0:       # contract left to right
       mps1_ = mps1
       mpo_  = mpo
       mps2_ = mps2
    elif arrow == 1:       # contract right to left
       mps1_ = mps1[::-1]
       mpo_  = mpo[::-1]
       mps2_ = mps2[::-1]
    else:
       print '[inprod fct]: ???'
   
    in_site = np.einsum('lnr,anNb,LNR->rbR',mps1[0],mpo[0],mps2[0])
    for i in range(1,L):
        in_site = np.einsum('rbR,lnr,anNb,LNR',in_site,mps1[i],mpo[i],mps2[i])

    assert(np.all(in_site == 1)), '[inprod fct: output not scalar'
    return in_site.squeeze()


def gemv(mpo,mps1,alpha=1,beta=1,mps2=None):
   # returns alpha* mpo1 mps1 + beta mps2 
   
   L = len(mps1)
   assert(len(mpo )==L), '[gemv: length of mps and mpo are not equal'
   assert(len(mps2)==L), '[gemv: length of mps and mpo are not equal'

   if mps2 is None:
      new_mps = np.empty(L,dtype=np.object)
      for i in range(L):
          new_mps[i] = np.einsum('anNb,lnr->alNbr',mpo,mps1)


