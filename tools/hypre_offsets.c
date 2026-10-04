/* The offsets of the few fields of hypre's matrix and vector structures that the
 * single-precision solver of hIPPyMFEM reads through ctypes, printed as JSON.  Compiled
 * against the headers and the HYPRE_config.h of the very build of the library. */
#include <stdio.h>
#include <stddef.h>
#include "_hypre_parcsr_mv.h"

int main(void)
{
   printf("{\n");
   printf(" \"version\": \"%s\",\n", HYPRE_RELEASE_VERSION);
   printf(" \"sizeof_int\": %d, \"sizeof_bigint\": %d, \"sizeof_real\": %d, \"sizeof_complex\": %d,\n",
          (int) sizeof(HYPRE_Int), (int) sizeof(HYPRE_BigInt), (int) sizeof(HYPRE_Real), (int) sizeof(HYPRE_Complex));
   printf(" \"par_diag\": %d, \"par_offd\": %d, \"par_col_map_offd\": %d, \"par_comm_pkg\": %d,\n",
          (int) offsetof(hypre_ParCSRMatrix, diag), (int) offsetof(hypre_ParCSRMatrix, offd),
          (int) offsetof(hypre_ParCSRMatrix, col_map_offd), (int) offsetof(hypre_ParCSRMatrix, comm_pkg));
   printf(" \"par_num_nonzeros\": %d, \"par_d_num_nonzeros\": %d,\n",
          (int) offsetof(hypre_ParCSRMatrix, num_nonzeros), (int) offsetof(hypre_ParCSRMatrix, d_num_nonzeros));
   printf(" \"csr_i\": %d, \"csr_j\": %d, \"csr_data\": %d, \"csr_num_rows\": %d, \"csr_num_cols\": %d, "
          "\"csr_num_nonzeros\": %d, \"csr_memory_location\": %d,\n",
          (int) offsetof(hypre_CSRMatrix, i), (int) offsetof(hypre_CSRMatrix, j),
          (int) offsetof(hypre_CSRMatrix, data), (int) offsetof(hypre_CSRMatrix, num_rows),
          (int) offsetof(hypre_CSRMatrix, num_cols), (int) offsetof(hypre_CSRMatrix, num_nonzeros),
          (int) offsetof(hypre_CSRMatrix, memory_location));
   printf(" \"parvec_local\": %d, \"vec_data\": %d, \"vec_size\": %d, \"parvec_all_zeros\": %d\n",
          (int) offsetof(hypre_ParVector, local_vector), (int) offsetof(hypre_Vector, data),
          (int) offsetof(hypre_Vector, size), (int) offsetof(hypre_ParVector, all_zeros));
   printf("}\n");
   return 0;
}
