#include <stdlib.h>
#include <stdio.h>

// Matrix-free matvec Ap = A*p for the CG solver's 2D 5-point Poisson stencil
// (diagonal 4, four neighbors -1), same operator as
// 2_mdh_sparse/mdh_generator_source/spec/cg_matvec.cpp -- fed to PPCG instead
// of the MDH generator, to compare the two auto-generation strategies.
//
// P is zero-padded (halo of 1 on each side) so the stencil access is a plain
// affine array reference with no boundary conditionals -- required for PPCG's
// polyhedral analysis, and mathematically equivalent to MDH's oob::ZERO
// (a missing neighbor's coefficient never appears in kernel_sparse.cu's CSR
// matrix, exactly like reading a zero-padded halo).

#define M 64

int main()
{
	float P[M + 2][M + 2];
	float Ap[M][M];
	int i, j;

	for (i = 0; i < M + 2; ++i)
		for (j = 0; j < M + 2; ++j)
			P[i][j] = 0.0f;

	for (i = 1; i <= M; ++i)
		for (j = 1; j <= M; ++j)
			P[i][j] = (float)((i * 7 + j * 13) % 11) - 5.0f;

#pragma scop
	for (i = 0; i < M; ++i) {
		for (j = 0; j < M; ++j) {
			Ap[i][j] = 4.0f * P[i + 1][j + 1]
			         - P[i][j + 1] - P[i + 2][j + 1]
			         - P[i + 1][j] - P[i + 1][j + 2];
		}
	}
#pragma endscop

	printf("%f\n", Ap[0][0]);
	return 0;
}
