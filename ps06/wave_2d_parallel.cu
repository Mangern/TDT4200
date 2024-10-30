#define _XOPEN_SOURCE 700 // hope its okay i changed this to 700 instead of 600
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <sys/time.h>

// TASK: T1
// Include the cooperative groups library
// BEGIN: T1
#include <cuda_runtime.h> // for my LSP
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
// END: T1


// Convert 'struct timeval' into seconds in double prec. floating point
#define WALLTIME(t) ((double)(t).tv_sec + 1e-6 * (double)(t).tv_usec)

// Option to change numerical precision
typedef int64_t int_t;
typedef float real_t;

// TASK: T1b
// Variables needed for implementation
// BEGIN: T1b

// somewhat helpful macros
//#define DEBUG
#ifdef DEBUG
#define INFO(...) printf(__VA_ARGS__)
#else
#define INFO(...) do {} while (0);
#endif

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

constexpr size_t SHARED_MEM_SIZE = (BLOCK_SIZE_X + 2) * (BLOCK_SIZE_Y + 2) * sizeof(real_t);

// Simulation parameters: size, step count, and how often to save the state
const int_t
    N = 128,
    M = 128,
    max_iteration = 1000000,
    snapshot_freq = 1000;

// Wave equation parameters, time step is derived from the space step
const real_t
    c  = 1.0,
    dx = 1.0,
    dy = 1.0;
real_t
    dt;

dim3 gridDims = {M / BLOCK_SIZE_X, N / BLOCK_SIZE_Y, 1};
dim3 blockDims = {BLOCK_SIZE_X, BLOCK_SIZE_Y, 1};

// Buffers for three time steps, indexed with 2 ghost points for the boundary_condition
// to be allocated on the device
real_t
    *d_buffers[3] = { NULL, NULL, NULL };

// buffer to copy to host in domain_save
real_t *h_buffer;

#define U_host(i, j) h_buffer[((i)+1)*(N+2)+(j)+1]

// instead of using U macros, this is more versatile
#define INDEX(i, j, WIDTH) (((i)+1) * ((WIDTH) + 2) + (j) + 1)
// END: T1b

#define cudaErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


// Rotate the time step buffers.
__device__
void move_buffer_window ( real_t** d_buffers0, real_t** d_buffers1, real_t** d_buffers2 )
{
    real_t *temp = *d_buffers0;
    *d_buffers0 = *d_buffers1;
    *d_buffers1 = *d_buffers2;
    *d_buffers2 = temp;
}


// Save the present time step in a numbered file under 'data/'
void domain_save ( int_t step )
{
    cudaErrorCheck(cudaMemcpy(h_buffer, d_buffers[1], (N+2) * (M+2) * sizeof(real_t), cudaMemcpyDeviceToHost));

    char filename[256];
    sprintf ( filename, "data/%.5ld.dat", step );
    FILE *out = fopen ( filename, "wb" );
    for ( int_t i=0; i<M; i++ )
    {
        fwrite ( &U_host(i,0), sizeof(real_t), N, out );
    }
    fclose ( out );
}


// TASK: T4
// Get rid of all the memory allocations
void domain_finalize ( void )
{
// BEGIN: T4
    if (h_buffer)
        free(h_buffer);

    for (int i = 0; i < 3; ++i)
        if (d_buffers[i])
            cudaErrorCheck(cudaFree(d_buffers[i]));
// END: T4
}


// TASK: T6
// Neumann (reflective) boundary condition
// Due to the way I handled time_step with shared memory
// I chose to inline boundary condition computations, 
// which means that this function is never called.
// BEGIN: T6
__device__
void boundary_condition ( real_t* d_buffer_cur )
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (j == 0) {
        d_buffer_cur[INDEX(i, -1, N)] = d_buffer_cur[INDEX(i, 1, N)];
    }
    if (j == N - 1) {
        d_buffer_cur[INDEX(i, N, N)]  = d_buffer_cur[INDEX(i, N-2, N)];
    }
    if (i == 0) {
        d_buffer_cur[INDEX(-1, j, N)] = d_buffer_cur[INDEX(1, j, N)];
    }
    if (i == M - 1) {
        d_buffer_cur[INDEX(M, j, N)]  = d_buffer_cur[INDEX(M-2, j, N)];
    }
}
// END: T6


// TASK: T5
// Integration formula
// BEGIN; T5
__device__
void time_step ( real_t dt, real_t* d_buffer_prv, real_t* d_buffer_cur, real_t* d_buffer_nxt )
{
    // We access U quite a lot in the calculation, so store all values in 
    // this shared buffer first.
    __shared__ real_t s_U[(BLOCK_SIZE_X + 2) * (BLOCK_SIZE_Y + 2)];

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    int local_i = threadIdx.y;
    int local_j = threadIdx.x;

    // populate shared memory
    s_U[INDEX(local_i, local_j, BLOCK_SIZE_X)] = d_buffer_cur[INDEX(i, j, N)];

    // threads on the border have to copy from 'outside' this block
    // top border
    if (local_i == 0) {
        // inline boundary condition
        if (i == 0) s_U[INDEX(-1, local_j, BLOCK_SIZE_X)] = d_buffer_cur[INDEX(1, j, N)];
        else        s_U[INDEX(-1, local_j, BLOCK_SIZE_X)] = d_buffer_cur[INDEX(i-1, j, N)];
    }
    // bottom border
    if (local_i == BLOCK_SIZE_Y - 1) {
        // inline boundary condition
        if (i == M - 1) s_U[INDEX(BLOCK_SIZE_Y, local_j, BLOCK_SIZE_X)] = d_buffer_cur[INDEX(M - 2, j, N)];
        else            s_U[INDEX(BLOCK_SIZE_Y, local_j, BLOCK_SIZE_X)] = d_buffer_cur[INDEX(i+1, j, N)];
    }

    // left border
    if (local_j == 0) {
        // inline boundary condition
        if (j == 0) s_U[INDEX(local_i, -1, BLOCK_SIZE_X)] = d_buffer_cur[INDEX(i, 1, N)];
        else        s_U[INDEX(local_i, -1, BLOCK_SIZE_X)] = d_buffer_cur[INDEX(i, j - 1, N)];
    }

    // right border
    if (local_j == BLOCK_SIZE_X - 1) {
        // inline boundary condition
        if (j == N - 1) s_U[INDEX(local_i, BLOCK_SIZE_X, BLOCK_SIZE_X)] = d_buffer_cur[INDEX(i, N - 2, N)];
        else            s_U[INDEX(local_i, BLOCK_SIZE_X, BLOCK_SIZE_X)] = d_buffer_cur[INDEX(i, j + 1, N)];
    }

    // Sync to avoid race
    __syncthreads();

    // Do the actual forward pass from shared memory
    d_buffer_nxt[INDEX(i,j, N)] = -d_buffer_prv[INDEX(i,j,N)] 
          + 2.0*s_U[INDEX(local_i,local_j, BLOCK_SIZE_X)]
             + (dt*dt*c*c)/(dx*dy) * (
                s_U[INDEX(local_i-1, local_j, BLOCK_SIZE_X)]
              + s_U[INDEX(local_i+1, local_j, BLOCK_SIZE_X)]
              + s_U[INDEX(local_i, local_j-1, BLOCK_SIZE_X)]
              + s_U[INDEX(local_i, local_j+1, BLOCK_SIZE_X)]
              -4.0 * s_U[INDEX(local_i, local_j, BLOCK_SIZE_X)]
             );
}
// END: T5

__global__
void simulate_iterations ( real_t dt, real_t* d_buffer_prv, real_t* d_buffer_cur, real_t* d_buffer_nxt )
{
    for (int iteration = 0; iteration < snapshot_freq; ++iteration) {
        // sync everything
        cg::this_grid().sync();

        // Derive step t+1 from steps t and t-1
        time_step(dt, d_buffer_prv, d_buffer_cur, d_buffer_nxt);

        // Rotate the time step buffers
        move_buffer_window(&d_buffer_prv, &d_buffer_cur, &d_buffer_nxt);
    }
}


// TASK: T7
// Main time integration.
void simulate( void )
{
// BEGIN: T7
    // Go through each time step
    for ( int_t iteration=0; iteration<=max_iteration; iteration += snapshot_freq )
    {
        domain_save ( iteration / snapshot_freq );

        // Launch cooperative kernels
        void* args[] = {&dt, &d_buffers[0], &d_buffers[1], &d_buffers[2]};
        cudaErrorCheck(cudaLaunchCooperativeKernel((void*)simulate_iterations, gridDims, blockDims, args, SHARED_MEM_SIZE));

        // to keep in sync with the device, swap on host as well
        for (int i = 0; i < snapshot_freq % 3; ++i) {
            real_t *temp = d_buffers[0];
            d_buffers[0] = d_buffers[1];
            d_buffers[1] = d_buffers[2];
            d_buffers[2] = temp;
        }
    }
// END: T7
}


// TASK: T8
// GPU occupancy
void occupancy( void )
{
// BEGIN: T8
    cudaDeviceProp deviceProp;
    cudaErrorCheck(cudaGetDeviceProperties(&deviceProp, 0));
    
    int maxActiveBlocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, (void *)simulate_iterations, BLOCK_SIZE_X * BLOCK_SIZE_Y, SHARED_MEM_SIZE);

    int maxWarpsPerMultiprocessor = deviceProp.maxThreadsPerMultiProcessor / deviceProp.warpSize;
    int activeWarps = maxActiveBlocks * (BLOCK_SIZE_X * BLOCK_SIZE_Y / deviceProp.warpSize);

    float occupancy = (float)activeWarps / maxWarpsPerMultiprocessor;

    printf("Grid size set to:             %d.\n", gridDims.x * gridDims.y * gridDims.z);
    printf("Launched blocks of size:      %d.\n", blockDims.x * blockDims.y * blockDims.z);
    printf("Active warps:                 %d.\n", activeWarps);
    printf("Max warps per multiprocessor: %d.\n", maxWarpsPerMultiprocessor);
    printf("Theoretical occupancy:        %f.\n", occupancy);
    printf("\n");
// END: T8
}


// TASK: T2
// Make sure at least one CUDA-capable device exists
static bool init_cuda()
{
// BEGIN: T2
    int count;
    cudaErrorCheck(cudaGetDeviceCount(&count));
    printf("CUDA device count: %d\n", count);
    if (count == 0) {
        fprintf(stderr, "No CUDA-compatible device found.\n");
        return 0;
    }

    cudaErrorCheck(cudaSetDevice(0));

    cudaDeviceProp deviceProp;
    cudaErrorCheck(cudaGetDeviceProperties(&deviceProp, 0));

    printf("Device name: %s\n", deviceProp.name);
    printf("Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("Multiprocessors: %d\n", deviceProp.multiProcessorCount);
    printf("Warp size: %d\n", deviceProp.warpSize);
    printf("Global memory: %.1fGiB\n", (float)deviceProp.totalGlobalMem / float(1<<30));
    printf("Per-block shared memory: %.1fkiB\n", (float)deviceProp.sharedMemPerBlock / float(1<<10));
    printf("Per-block registers: %d\n", deviceProp.regsPerBlock);

    return 1;
// END: T2
}


// TASK: T3
// Set up our three buffers, and fill two with an initial perturbation
void domain_initialize ( void )
{
// BEGIN: T3
    bool locate_cuda = init_cuda();
    if (!locate_cuda)
    {
        exit( EXIT_FAILURE );
    }

    // Allocate the needed buffers
    // host
    h_buffer = (real_t *)malloc((M+2) * (N+2) * sizeof(real_t));
    // device
    cudaErrorCheck(cudaMalloc(&d_buffers[0], (M+2) * (N+2) * sizeof(real_t)));
    cudaErrorCheck(cudaMalloc(&d_buffers[1], (M+2) * (N+2) * sizeof(real_t)));
    cudaErrorCheck(cudaMalloc(&d_buffers[2], (M+2) * (N+2) * sizeof(real_t)));

    // calculate initial condition
    for ( int_t i=0; i<M; i++ )
    {
        for ( int_t j=0; j<N; j++ )
        {
            // Calculate delta (radial distance) adjusted for M x N grid
            real_t delta = sqrt ( ((i - M/2.0) * (i - M/2.0)) / (real_t)M +
                                ((j - N/2.0) * (j - N/2.0)) / (real_t)N );
            U_host(i, j) = exp ( -4.0*delta*delta );
        }
    }

    // transfer initial condition to device
    cudaMemcpy(d_buffers[1], h_buffer,     (N+2) * (M+2) * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_buffers[0], d_buffers[1], (N+2) * (M+2) * sizeof(real_t), cudaMemcpyDeviceToDevice);

    // Set the time step for 2D case
    dt = dx*dy / (c * sqrt (dx*dx+dy*dy));

    INFO("Initialized\n");
// END: T3
}


int main ( void )
{
    // Set up the initial state of the domain
    domain_initialize();

    struct timeval t_start, t_end;

    gettimeofday ( &t_start, NULL );
    simulate();
    gettimeofday ( &t_end, NULL );

    printf ( "Total elapsed time: %lf seconds\n",
        WALLTIME(t_end) - WALLTIME(t_start)
    );

    occupancy();

    // Clean up and shut down
    domain_finalize();
    exit ( EXIT_SUCCESS );
}
