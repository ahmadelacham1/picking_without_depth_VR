#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define INF 999999.0

// Structure to represent a point in the alignment path
typedef struct {
    int i;
    int j;
} PathPoint;

void dtw(double *x, int N, double *y, int M, double *alignment_cost, PathPoint **alignment_path, int *path_length) {
    double **dist, **cost;
    int i, j;

    // Allocate memory for distance matrix
    double **dist_mat = (double **)malloc(N * sizeof(double *));
    for (i = 0; i < N; i++) {
        dist_mat[i] = (double *)malloc(M * sizeof(double));
    }

    // Compute distance matrix
    dist = dist_mat;
    for (i = 0; i < N; i++) {
        for (j = 0; j < M; j++) {
            dist[i][j] = fabs(x[i] - y[j]);
        }
    }

    // Allocate memory for cost matrix
    double **cost_mat = (double **)malloc((N + 1) * sizeof(double *));
    for (i = 0; i < N + 1; i++) {
        cost_mat[i] = (double *)malloc((M + 1) * sizeof(double));
    }

    // Initialize cost matrix
    cost = cost_mat;
    for (i = 0; i <= N; i++) {
        cost[i][0] = INF;
    }
    for (j = 0; j <= M; j++) {
        cost[0][j] = INF;
    }
    cost[0][0] = 0.0;

    // Fill cost matrix and find traceback path
    for (i = 1; i <= N; i++) {
        for (j = 1; j <= M; j++) {
            double penalty[3] = {
                cost[i - 1][j - 1],   // match (0)
                cost[i - 1][j],       // insertion (1)
                cost[i][j - 1]        // deletion (2)
            };
            int i_penalty = 0;
            if (penalty[1] < penalty[i_penalty]) {
                i_penalty = 1;
            }
            if (penalty[2] < penalty[i_penalty]) {
                i_penalty = 2;
            }
            cost[i][j] = dist[i - 1][j - 1] + penalty[i_penalty];
        }
    }

    // Traceback from bottom right to top left
    i = N;
    j = M;
    *alignment_cost = 0.0;

    // Calculate path length
    *path_length = N;
    *alignment_path = (PathPoint *)malloc(N * sizeof(PathPoint));

    // Open a file to write the alignment path in JSON format
    FILE *path_file = fopen("alignment_path.json", "w");
    if (path_file == NULL) {
        fprintf(stderr, "Error opening path file.\n");
        exit(1);
    }

    fprintf(path_file, "[");

    while (i > 0 && j > 0) {
        (*alignment_path)[N - i].i = i - 1;
        (*alignment_path)[N - i].j = j - 1;

        *alignment_cost += dist[i - 1][j - 1];

        fprintf(path_file, "[%d, %d]", i - 1, j - 1);

        if (i > 1 || j > 1) {
            fprintf(path_file, ", ");
        }

        double penalty[3] = {
            cost[i - 1][j - 1],   // match (0)
            cost[i - 1][j],       // insertion (1)
            cost[i][j - 1]        // deletion (2)
        };
        int i_penalty = 0;
        if (penalty[1] < penalty[i_penalty]) {
            i_penalty = 1;
        }
        if (penalty[2] < penalty[i_penalty]) {
            i_penalty = 2;
        }
        if (i_penalty == 0) {
            i--;
            j--;
        } else if (i_penalty == 1) {
            i--;
        } else {
            j--;
        }
    }

    fprintf(path_file, "]");
    fclose(path_file);
}

int main() {
    // Example data
    int N = 100;
    int M = 100;
    double alignment_cost;

    double *x = (double *)malloc(N * sizeof(double));
    double *y = (double *)malloc(M * sizeof(double));

    for (int i = 0; i < N; i++) {
        double idx = 6.28 * i / (N - 1);
        x[i] = sin(idx) + ((double)rand() / RAND_MAX - 0.5) / 10.0;
    }

    for (int i = 0; i < M; i++) {
        double idx = 6.28 * i / (M - 1);
        y[i] = cos(idx);
    }

    // Define pointers for the alignment path
    PathPoint *alignment_path;
    int path_length;

    // Call the dtw function with alignment path
    dtw(x, N, y, M, &alignment_cost, &alignment_path, &path_length);

    // Print the alignment cost
    printf("Alignment cost: %.4f\n", alignment_cost);

    // Print the alignment path
    printf("Alignment path:\n");
    for (int k = 0; k < path_length; k++) {
        printf("(%d, %d) ", alignment_path[k].i, alignment_path[k].j);
    }
    printf("\n");

    // Free allocated memory
    free(x);
    free(y);
    // Free memory for the alignment path
    free(alignment_path);
    return 0;
}
