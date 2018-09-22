#include <stdio.h>
#include "mkl.h"

extern "C" void spmm(const int & M, const int & N, const int & K, 
		int* rowIndex_A, int* columns_A, double* values_A,
		int* rowIndex_B, int* columns_B, double* values_B,
		int*& pointerB_C, int*& pointerE_C, int*& columns_C, double*& values_C, int & nnz, 
		sparse_matrix_t & handle_C_p){

	/* 
	 * Calculate C = A * B, where shape of A is M * K, B is K * N.
	 * All matrices are in CSR format, i.e. can be expressed with (compressed)rowIndex, columns and values.
	 * where length of rowIndex is number of row + 1, while length columns and values is number of non-zero elements,
	 * nnz.
	 * pointerB_C / pointerE_C is collection of begin / end index of each row of C.
	 *
	 * Since we usually do NOT know nnz of C before the calculation, we return it as well.
	 * 
	 * See https://software.intel.com/en-us/mkl-developer-reference-c-sparse-blas-csr-matrix-storage-format
	 * for details.
	 *
	*/

	int  rows, cols;
	sparse_index_base_t    indexing;
	sparse_matrix_t        csrA = NULL, csrB = NULL, csrC = NULL;

	// create CSR handle of A and B
	mkl_sparse_d_create_csr( &csrA, SPARSE_INDEX_BASE_ZERO, M, K, rowIndex_A, rowIndex_A+1, columns_A, values_A );
	mkl_sparse_d_create_csr( &csrB, SPARSE_INDEX_BASE_ZERO, K, N, rowIndex_B, rowIndex_B+1, columns_B, values_B );

	//mkl_sparse_set_memory_hint (csrA, SPARSE_MEMORY_AGGRESSIVE);
	//mkl_sparse_set_memory_hint (csrB, SPARSE_MEMORY_AGGRESSIVE);
	mkl_sparse_optimize( csrA );
	mkl_sparse_optimize( csrB );

	// do multiplication and export the info.
	mkl_sparse_spmm( SPARSE_OPERATION_NON_TRANSPOSE, csrA, csrB, &csrC );
	mkl_sparse_d_export_csr( csrC, &indexing, &rows, &cols, &pointerB_C, &pointerE_C, &columns_C, &values_C );
	nnz = pointerE_C[M - 1];


	//destory the handlde of A and B
	mkl_sparse_destroy( csrA );
	mkl_sparse_destroy( csrB );
	handle_C_p = csrC; // handle C will be destoryed latter.
	//mkl_sparse_destroy( csrc );
}


extern "C" void free_handle(sparse_matrix_t & handle){
	mkl_sparse_destroy(handle);
}
