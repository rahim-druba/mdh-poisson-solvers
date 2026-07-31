#include <stdlib.h>
#include <stdio.h>
#define R 32
#define C 32
int main()
{
	float P[R + 2][C + 2];
	float Ap[R][C];
	int i, j;
	for (i = 0; i < R + 2; ++i)
		for (j = 0; j < C + 2; ++j)
			P[i][j] = 0.0f;
	for (i = 1; i <= R; ++i)
		for (j = 1; j <= C; ++j)
			P[i][j] = (float)((i * 7 + j * 13) % 11) - 5.0f;
#pragma scop
	for (i = 0; i < R; ++i) {
		for (j = 0; j < C; ++j) {
			Ap[i][j] = 4.0f * P[i + 1][j + 1]
			         - P[i][j + 1] - P[i + 2][j + 1]
			         - P[i + 1][j] - P[i + 1][j + 2];
		}
	}
#pragma endscop
	printf("%f\n", Ap[0][0]);
	return 0;
}
