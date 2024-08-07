#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

# define N 100
# define DEGREE 5
# define LEN (DEGREE + 1)
# define M LEN //  for extra.c
# define RAND_FLOAT (double)rand()/(double)RAND_MAX
# define RAND_INT rand() % 10

#include "./extra.c"


struct train_set {
    double X[N];
    double Y[N];
};

typedef double Weights[LEN];

void random_weights(Weights* out) {
    for (int i = 0; i<LEN; i++) {
        (*out)[i] = RAND_FLOAT;
    }
}

double f(Weights ws, double x) {
    double sum = 0;
    for (int i = 0; i<LEN; i++) {
        sum += pow(x, i) * ws[i];
    }
    return sum;
}

int main(void) {
    srand( time(NULL));
    
    Weights real = {0};
    random_weights(&real);
    printf("weights for real: {");
    for (int i = 0; i<LEN; i++) {
        printf("%fx^%d,", real[i], i);
    }
    printf("}\n");

    struct train_set train;
    for (int i = 0; i<N;i++) {
        train.X[i] = RAND_INT;
        train.Y[i] = f(real, train.X[i]);
    }

    Weights to_fit = {0};
    random_weights(&to_fit);
        
    double A[M][M] = {0};

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            A[i][j] = 0;
            for (int x = 0; x < N; x++) {
                A[i][j] += pow(train.X[x], j + i);
            }
        }
    }

    double inv_A[M][M] = {0};

    printf("\n\n");
    printMatrix(A, M, M);
    printf("\n\n");
    
    invertMatrix(A, inv_A);

    printf("to inv");
    printf("\n\n");
    printMatrix(inv_A, M, M);
    printf("\n\n");

    double Y_sum[M] = {0};
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            Y_sum[i] += train.Y[j] * pow(train.X[j], i);
        }
    }
    printf("\n\n");
    printVec(Y_sum);
    printf("\n\n");

    double res[M] = {0};

    matrixVectorMultiply(inv_A, Y_sum, res);
    
    printf("real: {");
    for (int i = 0; i < M; i++) {
        printf("%f, ", real[i]);
    }
    printf("}\n");

    printf("regressor: {");
    for (int i = 0; i < M; i++) {
        printf("%f, ", res[i]);
    }
    printf("}\n");
}   
