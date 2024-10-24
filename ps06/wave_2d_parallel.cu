#include <cassert>
#define _XOPEN_SOURCE 600
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
#define DEBUG
#undef DEBUG
#ifdef DEBUG
#define INFO(...) printf(__VA_ARGS__)
#else
#define INFO(...) do {} while (0);
#endif

#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 32

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

real_t *h_buffer;

#define U_prv(i,j) d_buffers0[((i)+1)*(N+2)+(j)+1]
#define U(i,j)     d_buffers1[((i)+1)*(N+2)+(j)+1]
#define U_nxt(i,j) d_buffers2[((i)+1)*(N+2)+(j)+1]

#define U_host(i, j) h_buffer[((i)+1)*(N+2)+(j)+1]
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
void move_buffer_window ( void )
{
    real_t *temp = d_buffers[0];
    d_buffers[0] = d_buffers[1];
    d_buffers[1] = d_buffers[2];
    d_buffers[2] = temp;
}


// Save the present time step in a numbered file under 'data/'
void domain_save ( int_t step )
{
    INFO("%p %p %p\n", d_buffers[0], d_buffers[1], d_buffers[2]);
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
// BEGIN: T6
__device__
void boundary_condition ( real_t* d_buffers0, real_t* d_buffers1, real_t* d_buffers2 )
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (j == 0) {
        U(i,-1) = U(i,1);
    }
    if (j == N - 1) {
        U(i,N)  = U(i,N-2);
    }
    if (i == 0) {
        U(-1,j) = U(1,j);
    }
    if (i == M - 1) {
        U(M,j)  = U(M-2,j);
    }
}
// END: T6


// TASK: T5
// Integration formula
// BEGIN; T5
__global__
void time_step ( real_t dt, real_t* d_buffers0, real_t* d_buffers1, real_t* d_buffers2 )
{
    boundary_condition(d_buffers0, d_buffers1, d_buffers2);
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    //for ( int_t i=0; i<M; i++ )
    //{
    //    for ( int_t j=0; j<N; j++ )
    //    {
            U_nxt(i,j) = -U_prv(i,j) + 2.0*U(i,j)
                     + (dt*dt*c*c)/(dx*dy) * (
                        U(i-1,j)+U(i+1,j)+U(i,j-1)+U(i,j+1)-4.0*U(i,j)
                     );
    //    }
    //}
}
// END: T5


// TASK: T7
// Main time integration.
void simulate( void )
{
// BEGIN: T7
    // Go through each time step
    for ( int_t iteration=0; iteration<=max_iteration; iteration++ )
    {
        if ( (iteration % snapshot_freq)==0 )
        {
            cudaDeviceSynchronize();
            domain_save ( iteration / snapshot_freq );
        }

        cudaDeviceSynchronize();

        // Derive step t+1 from steps t and t-1
        //boundary_condition<<<gridDims, blockDims>>>();

        time_step<<<gridDims, blockDims>>>(dt, d_buffers[0], d_buffers[1], d_buffers[2]);

        cudaDeviceSynchronize();

        // Rotate the time step buffers
        move_buffer_window();
    }
// END: T7
}


// TASK: T8
// GPU occupancy
void occupancy( void )
{
// BEGIN: T8
    ;
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

    h_buffer = (real_t *)malloc((M+2) * (N+2) * sizeof(real_t));
    cudaErrorCheck(cudaMalloc(&d_buffers[0], (M+2) * (N+2) * sizeof(real_t)));
    cudaErrorCheck(cudaMalloc(&d_buffers[1], (M+2) * (N+2) * sizeof(real_t)));
    cudaErrorCheck(cudaMalloc(&d_buffers[2], (M+2) * (N+2) * sizeof(real_t)));

    INFO("%p %p %p\n", d_buffers[0], d_buffers[1], d_buffers[2]);

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
    // TODO: write kernel for applying initial condition on the device?
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
