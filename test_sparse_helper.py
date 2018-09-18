#! /usr/bin/env python

import operator

import numpy as np
import pytest
import scipy.sparse
import scipy.stats

import sparse
from sparse import COO
from sparse.utils import assert_eq, random_value_array
from sparse.coo import core
import sparse_helper as sh


@pytest.mark.parametrize('shape_x, shape_y, descr', [
    [(8,6), (8,4,6), "ij, klm -> ljkim"],
    [(1,2), (1,) , "ij, k -> jki"],
])
def test_einsum_outer_prod(shape_x, shape_y, descr):
   
    density = 0.4
    #np.random.seed(2)
    np.set_printoptions(3, linewidth = 1000, suppress = True)
    x = sparse.random(shape_x, density, format='coo')
    x_d = x.todense()
    y = sparse.random(shape_y, density, format='coo')
    y_d = y.todense()
    sout = sh.einsum(descr, x, y)
    out = np.einsum(descr, x_d, y_d)
    assert_eq(sout, out)

#def test_einsum_inner_contract(shape1, expr):
#   
#    density = 0.4
#    #np.random.seed(2)
#    np.set_printoptions(3, linewidth = 1000, suppress = True)
#    x = sparse.random(shape1, density, format='coo')
#    x_d = x.todense()
#    sout = sh.einsum(expr, x)
#    out = np.einsum(expr, x_d)
#    #assert(np.allclose(sout.todense(), out))
#    assert_eq(sout, out)


@pytest.mark.parametrize('shape_x, axes, descr', [
    [(8,8), None, "ii -> i"],
    [(3, 6, 6, 7, 6, 1), [1,2,4], 'ijjkjl -> ijkl'],
    [(1,2), [1], 'ij->ij'],
    [(5,), None, 'i->i'],
    [(7,7,21,7), [0,3], 'ijki->ijk']
])
def test_diagonal(shape_x, axes, descr):
    shape = (3, 6, 6, 7, 6, 1)
    density = 0.4
    #np.random.seed(2)
    np.set_printoptions(3, linewidth = 1000, suppress = True)
    x = sparse.random(shape_x, density, format='coo')
    x_d = x.todense()
    #y = sparse.random(shape2, density, format='coo')
    #y_d = y.todense()
    sout = sh.diagonal(x, axes = axes)

    #sout = sh.einsum(expr, x, y)
    out = np.einsum(descr, x_d)
    #assert(np.allclose(sout.todense(), out))
    assert_eq(sout, out)



if __name__ == '__main__':

    shape1 = (12, 13, 7)
    shape2 = (3, 2, 6, 7)
    expr = 'ijq, wpkl -> jwpklqi'
    test_einsum_outer_prod(shape1, shape2, expr)
    
    test_einsum_inner_contract(shape1, expr)

