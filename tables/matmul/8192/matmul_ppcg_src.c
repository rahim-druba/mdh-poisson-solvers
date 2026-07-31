#include <stdlib.h>
#include <stdio.h>
#define N 8192
int main()
{
	float A[N][N];
	float B[N][N];
	float S[N][N];
	int i, j, k;
	float buf;
	for (i = 0; i < N; ++i)
		for (k = 0; k < N; ++k)
			A[i][k] = (float)((i + k) % 5) - 2.0f;
	for (k = 0; k < N; ++k)
		for (j = 0; j < N; ++j)
			B[k][j] = (float)((k + 2 * j) % 5) - 2.0f;
#pragma scop
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			buf = 0.0f;
			for (k = 0; k < N; ++k) {
				buf += A[i][k] * B[k][j];
			}
			S[i][j] = buf;
		}
	}
#pragma endscop
	printf("%f\n", S[0][0]);
	return 0;
}
