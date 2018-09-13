import re
import numpy as np

def svd(idx, a, D=0, preserve_uv=None):
    idx0 = re.split(",", idx)
    assert len(idx0) == 2
    idx0[0].replace(" ", "")

    nsplit = len(idx0[0]) 

    a_shape = a.shape
    a = np.reshape(a, [np.prod(a.shape[:nsplit]), -1])

    u, s, vt = scipy.linalg.svd(a, full_matrices = False)
    
    M = len(s)
    if D > 0:
        M = min(D, M)

    dwt = np.sum(s[M:])
    u = u[:,:M]
    s = s[:M]
    vt = vt[:M,:]

    if preserve_uv == "u":
        ubig = np.zeros_like(a)
        ubig[:u.shape[0],:u.shape[1]] = u
        sbig = np.zeros((a.shape[1],),dtype=a.dtype)
        sbig[:s.shape[0]] = s
        vtbig = np.zeros((a.shape[1], a.shape[1]),dtype=a.dtype)
        vtbig[:vt.shape[0],:vt.shape[1]] = vt
        u, s, vt = ubig, sbig, vtbig
    elif preserve_uv == "v":
        vtbig = np.zeros_like(a)
        vtbig[:v.shape[0],:v.shape[1]] = v
        sbig = np.zeros((a.shape[0],),dtype=a.dtype)
        sbig[:s.shape[0]] = s
        ubig = np.zeros((a.shape[0], a.shape[0]),dtype=a.dtype)
        ubig[:u.shape[0],:u.shape[1]] = u
        u, s, vt = ubig, sbig, vtbig 
            
    u = np.reshape(u, (a_shape[:nsplit] + (-1,)))
    vt = np.reshape(vt, ((-1,) + a_shape[nsplit:]))

    return u, s, vt, dwt
