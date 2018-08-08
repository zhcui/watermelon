class MPX(object):
    """
    Lightweight MPS/MPO class
    """
    def __init__(self, mparray):
        self.mparray = mparray

    def __add__(self, other):
        return MPX(axpy(1., self.mparray, other.mparray))
    
    def __sub__(self, other):
        return MPX(axpby(1., self.mparray, -1., other.mparray))

    def __mul__(self, other):
        return MPX(dot(0, self.mparray, other.mparray))

    def __neg__(self):
        return MPX(scal(-1., self.mparray))
    
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return MPX(self.mparray.__array_ufunc__(ufunc, method, inputs, kwargs)

    def __str__(self):
        return str(self.mparray)
