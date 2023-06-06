#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#define S_LEN 512
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

typedef struct
{
    int val;
    int i;
    int j;
} max_ij;

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

void sw(int **sc_mat, char **dir_mat, char **query, char **reference, int *res, char **simple_rev_cigar)
{
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
}

__device__ void backtrace_gpu(char *d_dir_mat, char *d_simple_rev_cigar, int maxi, int maxj, const int blockShift, const int blockShift_dm)
{
    for (int n = 0; n < S_LEN * 2 && d_dir_mat[blockShift_dm + maxi * (S_LEN + 1) + maxj] != 0; n++)
    {
        int dir = d_dir_mat[blockShift_dm + maxi * (S_LEN + 1) + maxj];
        if (dir == 1 || dir == 2)
        {
            maxi--;
            maxj--;
        }
        else if (dir == 3)
            maxi--;
        else if (dir == 4)
            maxj--;

        d_simple_rev_cigar[blockShift * 2 + n] = dir;
    }
}

__global__ void sw_gpu(char *d_query, char *d_reference, char *d_dir_mat, int *d_res, char *d_simple_rev_cigar)
{
    unsigned int threadId = threadIdx.x;

    // I keep in shared memory only the last 2 computed scoring matrix diagonals + array of max structs for parallel reduction
    __shared__ int d_sc_last_d[S_LEN + 1];
    __shared__ int d_sc_2_to_last_d[S_LEN + 1];
    __shared__ max_ij max[S_LEN];

    int blockShift = blockIdx.x * S_LEN;
    int blockShift_dm = blockIdx.x * (S_LEN + 1) * (S_LEN + 1);

    // initialize the last 2 computed scoring matrix diagonals to 0 and the value field in the array of max structs to -2
    d_sc_last_d[threadId] = 0;
    d_sc_2_to_last_d[threadId] = 0;
    max[threadId].val = ins;
    if (threadId == 0)
    {
        d_sc_last_d[S_LEN] = 0;
        d_sc_2_to_last_d[S_LEN] = 0;
    }
    __syncthreads();

    // thread local variables
    int i, j, maxi, maxj, comparison, tmp1, tmp2, comparisonRes, delRes, insRes, tmp = 0;
    char dir;

    // loop for each diagonal of the scoring matrix
    for (int d = 0; d < S_LEN * 2 - 1; d++)
    {
        i = threadId + 1; // row index
        j = d - threadId + 1; // column index

        // set first row and first column of direction matrix to 0
        if (i == 0 || j == 0)
            d_dir_mat[blockShift_dm + i * (S_LEN + 1) + j] = 0;

        // check if indexes are valid
        if (!(i < 1 || i > S_LEN || j < 1 || j > S_LEN))
        {

            // compare the sequences characters
            comparison = (d_query[blockShift + i - 1] == d_reference[blockShift + j - 1]) ? match : mismatch;

            // compute the cell knowing the comparison result
            comparisonRes = d_sc_2_to_last_d[threadId] + comparison;
            delRes = d_sc_last_d[threadId] + del;
            insRes = d_sc_last_d[threadId + 1] + ins;

            tmp1 = comparisonRes > delRes ? comparisonRes : delRes;
            tmp2 = insRes > 0 ? insRes : 0;
            tmp = tmp1 > tmp2 ? tmp1 : tmp2;

            if (tmp == comparisonRes)
                dir = comparison == match ? 1 : 2;
            else if (tmp == delRes)
                dir = 3;
            else if (tmp == insRes)
                dir = 4;
            else
                dir = 0;

            // update direction matrix element (i, j)
            d_dir_mat[blockShift_dm + i * (S_LEN + 1) + j] = dir;

            // update local max of the thread
            if (max[threadId].val < tmp)
            {
                max[threadId].val = tmp;
                max[threadId].i = i;
                max[threadId].j = j;
            }
        }
        __syncthreads();

        // update last 2 computed diagonals
        d_sc_2_to_last_d[threadId + 1] = d_sc_last_d[threadId + 1];
        d_sc_last_d[threadId + 1] = tmp;
        __syncthreads();
    }

    // parallel reduction to find max
    for (int i = blockDim.x / 2; i > 0; i >>= 1)
    {
        if (threadId < i)
        {
            // I choose the same maximum found in the s-w implementation on the host (the first scanning the matrix row by row)
            if (max[threadId + i].val > max[threadId].val ||
                max[threadId + i].val == max[threadId].val && (max[threadId + i].i < max[threadId].i ||
                                                               (max[threadId + i].i == max[threadId].i && max[threadId + i].j < max[threadId].j)))
            {
                max[threadId] = max[threadId + i];
            }
        }
        __syncthreads();
    }

    // only the first thread for each block performs the backtrace and sets the results
    if (threadId == 0)
    {
        d_res[blockIdx.x] = max[0].val;

        maxi = max[0].i;
        maxj = max[0].j;
        backtrace_gpu(d_dir_mat, d_simple_rev_cigar, maxi, maxj, blockShift, blockShift_dm);
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
    char *query_copy = (char *)malloc(N * S_LEN * sizeof(char)); // for the GPU

    char **reference = (char **)malloc(N * sizeof(char *));
    for (int i = 0; i < N; i++)
        reference[i] = (char *)malloc(S_LEN * sizeof(char));
    char *reference_copy = (char *)malloc(N * S_LEN * sizeof(char)); // for the GPU

    int **sc_mat = (int **)malloc((S_LEN + 1) * sizeof(int *));
    for (int i = 0; i < (S_LEN + 1); i++)
        sc_mat[i] = (int *)malloc((S_LEN + 1) * sizeof(int));

    char **dir_mat = (char **)malloc((S_LEN + 1) * sizeof(char *));
    for (int i = 0; i < (S_LEN + 1); i++)
        dir_mat[i] = (char *)malloc((S_LEN + 1) * sizeof(char));

    int *res = (int *)malloc(N * sizeof(int));
    int *res_gpu = (int *)malloc(N * sizeof(int)); // for the result of GPU

    char **simple_rev_cigar = (char **)malloc(N * sizeof(char *));
    for (int i = 0; i < N; i++)
        simple_rev_cigar[i] = (char *)malloc(S_LEN * 2 * sizeof(char));
    char *simple_rev_cigar_gpu = (char *)malloc(N * S_LEN * 2 * sizeof(char)); // for the result of GPU

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
    char *d_query, *d_reference, *d_dir_mat, *d_simple_rev_cigar;
    int *d_res;

    CHECK(cudaMalloc(&d_query, N * S_LEN * sizeof(char)));
    CHECK(cudaMalloc(&d_reference, N * S_LEN * sizeof(char)));
    CHECK(cudaMalloc(&d_dir_mat, N * (S_LEN + 1) * (S_LEN + 1) * sizeof(char)));
    CHECK(cudaMalloc(&d_res, N * sizeof(int)));
    CHECK(cudaMalloc(&d_simple_rev_cigar, N * S_LEN * 2 * sizeof(char)));

    // CPU->GPU data transmission
    CHECK(cudaMemcpy(d_query, query_copy, sizeof(char) * N * S_LEN, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_reference, reference_copy, sizeof(char) * N * S_LEN, cudaMemcpyHostToDevice));

    // CPU execution
    double start_cpu = get_time();
    sw(sc_mat, dir_mat, query, reference, res, simple_rev_cigar);
    double end_cpu = get_time();

    // GPU execution
    double start_gpu = get_time();
    dim3 blocksPerGrid(N, 1, 1);
    dim3 threadsPerBlock(S_LEN, 1, 1);
    sw_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_query, d_reference, d_dir_mat, d_res, d_simple_rev_cigar);
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
    CHECK(cudaFree(d_dir_mat));
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