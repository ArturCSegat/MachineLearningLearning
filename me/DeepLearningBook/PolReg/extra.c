#include <stdio.h>
// #include <stdlib.h>

// #define M 4
// #define N 4

void invertMatrix(double mat[M][M], double inv[M][M]) {
    int i, j, k;
    double ratio;
    
    double augmented[M][2*M];
    for (i = 0; i < M; i++) {
        for (j = 0; j < M; j++) {
            augmented[i][j] = mat[i][j];
        }
        for (j = M; j < 2*M; j++) {
            if (i == j - M)
                augmented[i][j] = 1.0;
            else
                augmented[i][j] = 0.0;
        }
    }
    
    // Perform Gaussian elimination
    for (i = 0; i < M; i++) {
        if (augmented[i][i] == 0.0) {
            printf("Mathematical Error!\n");
            return;
        }
        for (j = 0; j < M; j++) {
            if (i != j) {
                ratio = augmented[j][i] / augmented[i][i];
                for (k = 0; k < 2*M; k++) {
                    augmented[j][k] -= ratio * augmented[i][k];
                }
            }
        }
    }

    // Row operation to make principal diagonal element to 1
    for (i = 0; i < M; i++) {
        for (j = M; j < 2*M; j++) {
            augmented[i][j] = augmented[i][j] / augmented[i][i];
        }
    }

    // Copy the right half of the augmented matrix to the inverse matrix
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            inv[i][j] = augmented[i][j + M];
        }
    }
}

void matrixVectorMultiply(double A[M][M], double x[M], double y[M]) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            y[i] += A[i][j] * x[j];
        }
    }
}

void printMatrix(double mat[M][M], int row, int col) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            printf("%f ", mat[i][j]);
        }
        printf("\n");
    }
}

void printVec(double mat[M]) {
    printf("{");
    for (int i = 0; i < M; i++) {
            printf("%f ", mat[i]);
    }
    printf("}\n");
}

// int main(void) {
//     double m[4][4] = {
//         {100.000000 ,431.000000 ,2615.000000 ,17867.000000             },
//         {431.000000 ,2615.000000 ,17867.000000 ,130871.000000          },
//         {2615.000000 ,17867.000000 ,130871.000000 ,1003451.000000      },
//         {17867.000000 ,130871.000000 ,1003451.000000 ,7944695.000000   },
//     };
//
//     double inv[4][4] = {0};
//
//
//     printf("initial m:\n");
//     printMatrix(m, 4, 4);
//     invertMatrix(m, inv);
//     printf("inverted m:\n");
//     printMatrix(inv, 4, 4);
//
// }
