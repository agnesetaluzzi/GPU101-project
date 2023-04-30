#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#define S_LEN 64
#define N 1000

// penalties
#define ins -2
#define del -2
#define match 1
#define mismatch -1

// error handling for CUDA API functions
#define CHECK(call)                                                  \
    {                                                                \
        const cudaError_t err = call;                                \
        if (err != cudaSuccess)                                      \
        {                                                            \
            printf("%s in %s at line %d\n", cudaGetErrorString(err), \
                   __FILE__, __LINE__);                              \
            exit(EXIT_FAILURE);                                      \
        }                                                            \
    }

// error handling for kernel invocations
#define CHECK_KERNELCALL()                                           \
    {                                                                \
        const cudaError_t err = cudaGetLastError();                  \
        if (err != cudaSuccess)                                      \
        {                                                            \
            printf("%s in %s at line %d\n", cudaGetErrorString(err), \
                   __FILE__, __LINE__);                              \
            exit(EXIT_FAILURE);                                      \
        }                                                            \
    }

double get_time() // function to get the time of day in seconds
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int max4(int n1, int n2, int n3, int n4)
{
    int tmp1, tmp2;
    tmp1 = n1 > n2 ? n1 : n2;
    tmp2 = n3 > n4 ? n3 : n4;
    tmp1 = tmp1 > tmp2 ? tmp1 : tmp2;
    return tmp1;
}

void backtrace(char *simple_rev_cigar, char **dir_mat, int i, int j, int max_cigar_len)
{
    int n;
    for (n = 0; n < max_cigar_len && dir_mat[i][j] != 0; n++)
    {
        int dir = dir_mat[i][j];
        if (dir == 1 || dir == 2)
        {
            i--;
            j--;
        }
        else if (dir == 3)
            i--;
        else if (dir == 4)
            j--;

        simple_rev_cigar[n] = dir;
    }
}

__global__ void kernel_main(char *d_query, char *d_reference, int *d_res, char *d_simple_rev_cigar)
{
    // int index = blockIdx.x * blockDim.x + threadIdx.x; // index of each element of the array
    __shared__ int d_sc_mat[S_LEN + 1][S_LEN + 1];
    __shared__ char d_dir_mat[S_LEN + 1][S_LEN + 1];
    int max = ins; // in sw all scores of the alignment are >= 0, so this will be for sure changed
    int maxi, maxj;
    // initialize the scoring matrix and direction matrix to 0
    for (int i = 0; i < S_LEN + 1; i++)
    {
        for (int j = 0; j < S_LEN + 1; j++)
        {
            d_sc_mat[i][j] = {0};
            d_dir_mat[i][j] = {0};
        }
    }

    for (int i = 1; i < S_LEN + 1; i++)
    {
        for (int j = 1; j < S_LEN + 1; j++)
        {
            // compare the sequences characters
            int comparison = (d_query[blockIdx.x * S_LEN + i - 1] == d_reference[blockIdx.x * S_LEN + j - 1]) ? match : mismatch;
            // compute the cell knowing the comparison result
            int tmp1, tmp2;
            tmp1 = (d_sc_mat[i - 1][j - 1] + comparison) > (d_sc_mat[i - 1][j] + del) ? (d_sc_mat[i - 1][j - 1] + comparison) : (d_sc_mat[i - 1][j] + del);
            tmp2 = (d_sc_mat[i][j - 1] + ins) > 0 ? (d_sc_mat[i][j - 1] + ins) : 0;
            int tmp = tmp1 > tmp2 ? tmp1 : tmp2;
            char dir;

            if (tmp == (d_sc_mat[i - 1][j - 1] + comparison))
                dir = comparison == match ? 1 : 2;
            else if (tmp == (d_sc_mat[i - 1][j] + del))
                dir = 3;
            else if (tmp == (d_sc_mat[i][j - 1] + ins))
                dir = 4;
            else
                dir = 0;

            d_dir_mat[i][j] = dir;
            d_sc_mat[i][j] = tmp;

            if (tmp > max)
            {
                max = tmp;
                maxi = i;
                maxj = j;
            }
        }
    }

    d_res[blockIdx.x] = d_sc_mat[maxi][maxj];

    int i = maxi;
    int j = maxj;
    for (int n = 0; n < S_LEN * 2 && d_dir_mat[i][j] != 0; n++)
    {
        int dir = d_dir_mat[i][j];
        if (dir == 1 || dir == 2)
        {
            i--;
            j--;
        }
        else if (dir == 3)
            i--;
        else if (dir == 4)
            j--;

        d_simple_rev_cigar[blockIdx.x * 2 * S_LEN + n] = dir;
    }
}

int main(int argc, char *argv[])
{
    srand(time(NULL));

    char alphabet[5] = {'A', 'C', 'G', 'T', 'N'};

    // host memory allocation
    char **query = (char **)malloc(N * sizeof(char *));
    for (int i = 0; i < N; i++)
        query[i] = (char *)malloc(S_LEN * sizeof(char));
    char *query_copy = (char *)malloc(N * S_LEN * sizeof(char));

    char **reference = (char **)malloc(N * sizeof(char *));
    for (int i = 0; i < N; i++)
        reference[i] = (char *)malloc(S_LEN * sizeof(char));
    char *reference_copy = (char *)malloc(N * S_LEN * sizeof(char));

    int **sc_mat = (int **)malloc((S_LEN + 1) * sizeof(int *));
    for (int i = 0; i < (S_LEN + 1); i++)
        sc_mat[i] = (int *)malloc((S_LEN + 1) * sizeof(int));
    char **dir_mat = (char **)malloc((S_LEN + 1) * sizeof(char *));
    for (int i = 0; i < (S_LEN + 1); i++)
        dir_mat[i] = (char *)malloc((S_LEN + 1) * sizeof(char));

    int *res = (int *)malloc(N * sizeof(int));
    int *res_gpu = (int *)malloc(N * sizeof(int));
    char **simple_rev_cigar = (char **)malloc(N * sizeof(char *));
    for (int i = 0; i < N; i++)
        simple_rev_cigar[i] = (char *)malloc(S_LEN * 2 * sizeof(char));
    char *simple_rev_cigar_gpu = (char *)malloc(N * S_LEN * 2 * sizeof(char));

    // randomly generate sequences
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < S_LEN; j++)
        {
            query[i][j] = alphabet[rand() % 5];
            query_copy[i * S_LEN + j] = query[i][j];

            reference[i][j] = alphabet[rand() % 5];
            reference_copy[i * S_LEN + j] = reference[i][j];
        }
    }

    // device memory allocation
    char *d_query, *d_reference, *d_simple_rev_cigar;
    int *d_res;

    CHECK(cudaMalloc(&d_query, N * S_LEN * sizeof(char)));

    CHECK(cudaMalloc(&d_reference, N * S_LEN * sizeof(char)));

    CHECK(cudaMalloc(&d_res, N * sizeof(int)));
    CHECK(cudaMalloc(&d_simple_rev_cigar, N * S_LEN * 2 * sizeof(char)));

    // CPU->GPU data transmission
    CHECK(cudaMemcpy(d_query, query_copy, sizeof(char) * N * S_LEN, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_reference, reference_copy, sizeof(char) * N * S_LEN, cudaMemcpyHostToDevice));

    // CPU execution
    double start_cpu = get_time();

    for (int n = 0; n < N; n++)
    {
        int max = ins; // in sw all scores of the alignment are >= 0, so this will be for sure changed
        int maxi, maxj;
        // initialize the scoring matrix and direction matrix to 0
        for (int i = 0; i < S_LEN + 1; i++)
        {
            for (int j = 0; j < S_LEN + 1; j++)
            {
                sc_mat[i][j] = 0;
                dir_mat[i][j] = 0;
            }
        }
        // compute the alignment
        for (int i = 1; i < S_LEN + 1; i++)
        {
            for (int j = 1; j < S_LEN + 1; j++)
            {
                // compare the sequences characters
                int comparison = (query[n][i - 1] == reference[n][j - 1]) ? match : mismatch;
                // compute the cell knowing the comparison result
                int tmp = max4(sc_mat[i - 1][j - 1] + comparison, sc_mat[i - 1][j] + del, sc_mat[i][j - 1] + ins, 0);
                char dir;

                if (tmp == (sc_mat[i - 1][j - 1] + comparison))
                    dir = comparison == match ? 1 : 2;
                else if (tmp == (sc_mat[i - 1][j] + del))
                    dir = 3;
                else if (tmp == (sc_mat[i][j - 1] + ins))
                    dir = 4;
                else
                    dir = 0;

                dir_mat[i][j] = dir;
                sc_mat[i][j] = tmp;

                if (tmp > max)
                {
                    max = tmp;
                    maxi = i;
                    maxj = j;
                }
            }
        }
        res[n] = sc_mat[maxi][maxj];
        backtrace(simple_rev_cigar[n], dir_mat, maxi, maxj, S_LEN * 2);
    }

    double end_cpu = get_time();

    // GPU execution
    double start_gpu = get_time();

    dim3 blocksPerGrid(N, 1, 1);
    dim3 threadsPerBlock(1, 1, 1);

    kernel_main<<<blocksPerGrid, threadsPerBlock>>>(d_query, d_reference, d_res, d_simple_rev_cigar);
    CHECK_KERNELCALL();

    CHECK(cudaDeviceSynchronize());

    double end_gpu = get_time();

    CHECK(cudaMemcpy(res_gpu, d_res, sizeof(int) * N, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(simple_rev_cigar_gpu, d_simple_rev_cigar, sizeof(char) * N * S_LEN * 2, cudaMemcpyDeviceToHost));

    for (int n = 0; n < N; n++)
    {
        if (res[n] != res_gpu[n])
        {
            fprintf(stderr, "ERRORE, RISULTATO SBAGLIATO SU GPU\n");
            break;
        }
        for (int s = 0; s < S_LEN * 2; s++)
        {
            if (simple_rev_cigar[n][s] != simple_rev_cigar_gpu[n * S_LEN * 2 + s])
            {
                fprintf(stderr, "ERRORE, RISULTATO SBAGLIATO SU GPU (BACKTRACE)\n");
                break;
            }
        }
    }

    printf("SW Time CPU: %.10lf\n", end_cpu - start_cpu);
    printf("SW Time GPU: %.10lf\n", end_gpu - start_gpu);

    CHECK(cudaFree(d_query));
    CHECK(cudaFree(d_reference));
    CHECK(cudaFree(d_res));
    CHECK(cudaFree(d_simple_rev_cigar));

    free(query);
    free(query_copy);
    free(reference);
    free(reference_copy);
    free(sc_mat);
    free(dir_mat);
    free(res);
    free(res_gpu);
    free(simple_rev_cigar);

    return 0;
}