#! /usr/bin/env bash
icpc -mkl -fPIC -shared -O3 spmm.c -o libspmm.so 
