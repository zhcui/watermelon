# Things to implement before version 1.0

# Big picture stuff

## Different Array libraries
* Infrastructure for inter-operability of array-like objects, see
  http://matthewrocklin.com/blog/work/2018/05/27/beyond-numpy
  and e.g. arrayish https://github.com/hameerabbasi/arrayish
  or custom implementation
  
* Interface to Cyclops Tensor Framework (modify python interface
  to be interoperable with pydata.sparse)
  * write paper with Edgar with large app
  
## Quantum chemistry
* Implement MPO for > 1 site dot algo and implement
  selected CI treatment of mat*vec
  * paper on selected CI + DMRG
  * many applications
* Check infrastructure for SU(2)
* Loop through improvements in QC
* Infinite DMRG with QC Hamiltonian

## PEPS
* implement 2D boundary contraction method; optimize sparse tensor performance

## Paper
  * paper on design of library and treatment of QN's, also
    part of documentation of v1.0
  
# Medium picture stuff

## Performance
* Optimize COO tensor performance

## DMRG
* Implement special DMRG version of MPO mult
* reduced density matrices

## MPS
* infinite MPS
* PBC Pippin gauge
* analytic derivative / tangent space 
* loop through all MPS time-evol algos
  * connect to Erika's work
  * maybe benchmark paper on time-evol algos, but others are doing
* MPS wrapper, mimic scipy.sparse iterative interface for Lanczos

## MPO
* Design interface for MPO generator based on
  Zhendong's MPO generator implementation

## Sparse implementation
* Finish Block COO library
* re-implement MPX functions with bCOO array

# Immediate TODO list and reminders

* Make pbc/obc DMRG code identical
* Implement Heisenberg with sparse: compare to dense
* Write quantum numbers to sparse map
  check preserved under svd
* check QN/Sparse map preserved under Heisenberg H
* Fix preserve_uv
* Fix rand constructor for complex
* Check out Zhendong MPO, implement basic QC H
* check quantum numbers/sparse map preserved
* Interface to randomized SVD
* general DMRG wrapper, to take scipy function e.g. solve_ivp
  