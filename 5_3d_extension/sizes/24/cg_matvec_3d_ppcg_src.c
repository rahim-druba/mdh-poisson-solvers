#include <stdlib.h>
#include <stdio.h>

// Matrix-free matvec Ap = A*p for the CG solver's 3D 7-point Poisson stencil
// (diagonal 6, six face-neighbors -1), same operator as
// mdh_cuda_stencil_kernels/kernels/cg_matvec_3d/spec/cg_matvec_3d.cpp -- fed
// to PPCG instead of the MDH generator, direct 3D extension of
// ../3_ppcg/cg_matvec_ppcg_src.c's 2D 5-point version.
//
// P is zero-padded (halo of 1 on each side, all 6 faces) so the stencil
// access is a plain affine array reference with no boundary conditionals --
// required for PPCG's polyhedral analysis, and mathematically equivalent to
// MDH's oob::ZERO.

#define M 24

int main()
{
	float P[M + 2][M + 2][M + 2];
	float Ap[M][M][M];
	int i, j, k;

	for (i = 0; i < M + 2; ++i)
		for (j = 0; j < M + 2; ++j)
			for (k = 0; k < M + 2; ++k)
				P[i][j][k] = 0.0f;

	for (i = 1; i <= M; ++i)
		for (j = 1; j <= M; ++j)
			for (k = 1; k <= M; ++k)
				P[i][j][k] = (float)((i * 7 + j * 13 + k * 17) % 11) - 5.0f;

#pragma scop
	for (i = 0; i < M; ++i) {
		for (j = 0; j < M; ++j) {
			for (k = 0; k < M; ++k) {
				Ap[i][j][k] = 6.0f * P[i + 1][j + 1][k + 1]
				            - P[i][j + 1][k + 1] - P[i + 2][j + 1][k + 1]
				            - P[i + 1][j][k + 1] - P[i + 1][j + 2][k + 1]
				            - P[i + 1][j + 1][k] - P[i + 1][j + 1][k + 2];
			}
		}
	}
#pragma endscop

	printf("%f\n", Ap[0][0][0]);
	return 0;
}
