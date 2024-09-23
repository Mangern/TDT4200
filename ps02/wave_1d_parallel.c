#define _XOPEN_SOURCE 600
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <sys/time.h>

// TASK: T1a
// Include the MPI headerfile
// BEGIN: T1a
#include <mpi.h>
// END: T1a
//
//#define DEBUG 1


// Option to change numerical precision.
typedef int64_t int_t;
typedef double real_t;


// TASK: T1b
// Declare variables each MPI process will need
// BEGIN: T1b
int world_size, world_rank;
// holds the size of every rank's sub-grids
int* grid_size;
// END: T1b


// Simulation parameters: size, step count, and how often to save the state.
const int_t
    N = 65536,
    max_iteration = 100000,
    snapshot_freq = 500;

// Wave equation parameters, time step is derived from the space step.
const real_t
    c  = 1.0,
    dx = 1.0;
real_t
    dt;

// Buffers for three time steps, indexed with 2 ghost points for the boundary.
real_t
    *buffers[3] = { NULL, NULL, NULL };


#define U_prv(i) buffers[0][(i)+1]
#define U(i)     buffers[1][(i)+1]
#define U_nxt(i) buffers[2][(i)+1]


// Convert 'struct timeval' into seconds in double prec. floating point
#define WALLTIME(t) ((double)(t).tv_sec + 1e-6 * (double)(t).tv_usec)


// TASK: T8
// Save the present time step in a numbered file under 'data/'.
void domain_save ( int_t step )
{
// BEGIN: T8
    // only root should save
    if (world_rank > 0) return;

    char filename[256];
    sprintf ( filename, "data/%.5ld.dat", step );
    FILE *out = fopen ( filename, "wb" );
    fwrite ( &U(0), sizeof(real_t), N, out );
    fclose ( out );
// END: T8
}


// TASK: T3
// Allocate space for each process' sub-grids
// Set up our three buffers, fill two with an initial cosine wave,
// and set the time step.
void domain_initialize ( void )
{
// BEGIN: T3

    // world_size is received from MPI_Comm_size
    grid_size = malloc(world_size * sizeof(int));

    // this will hold the index of the current rank's sub-grid start, 
    // for use in initial condition later
    int my_total_world_start = 0;

    for (int i = 0; i < world_size; ++i) {
        // sub-grid division may not be even, 
        // so this formula will make the first N % world_size have an ceil(N / world_size) and the rest floor
        grid_size[i] = N / world_size + (i < (N % world_size) ? 1 : 0);

        // start of current rank is sum of grid sizes of the ranks before it
        if (i < world_rank) {
            my_total_world_start += grid_size[i];
        }
    }

    int my_size = grid_size[world_rank];

    if (world_rank > 0) {
        buffers[0] = malloc ( (my_size+2)*sizeof(real_t) );
        buffers[1] = malloc ( (my_size+2)*sizeof(real_t) );
        buffers[2] = malloc ( (my_size+2)*sizeof(real_t) );
    } else {
        // if we are root, we need space for everything (for saving)
        // implementation wise it is easier if all three buffers are the same size
        // but we could save some space if N is large by only having one of them be that big.
        buffers[0] = malloc ( (N+2)*sizeof(real_t) );
        buffers[1] = malloc ( (N+2)*sizeof(real_t) );
        buffers[2] = malloc ( (N+2)*sizeof(real_t) );
    }

    for ( int_t i=0; i < my_size; i++ )
    {
        // total_index is i shifted to be the "global" location, so the cos wave is correct
        int total_index = my_total_world_start + i;
        U_prv(i) = U(i) = cos ( M_PI*total_index / (real_t)N );
    }

// END: T3

    // Set the time step for 1D case.
    dt = dx / c;
}


// Return the memory to the OS.
void domain_finalize ( void )
{
    free ( buffers[0] );
    free ( buffers[1] );
    free ( buffers[2] );
}


// Rotate the time step buffers.
void move_buffer_window ( void )
{
    real_t *temp = buffers[0];
    buffers[0] = buffers[1];
    buffers[1] = buffers[2];
    buffers[2] = temp;
}


// TASK: T4
// Derive step t+1 from steps t and t-1.
void time_step ( void )
{
// BEGIN: T4
    // mainly unchanged from sequential, but we don't need to go up to N anymore
    for ( int_t i=0; i < grid_size[world_rank]; i++ )
    {
        U_nxt(i) = -U_prv(i) + 2.0*U(i)
                 + (dt*dt*c*c)/(dx*dx) * (U(i-1)+U(i+1)-2.0*U(i));
    }
// END: T4
}


// TASK: T6
// Neumann (reflective) boundary condition.
void boundary_condition ( void )
{
// BEGIN: T6
    // only first and last rank needs to apply the boundary_condition
    if (world_rank == 0) {
        U(-1) = U(1);
    }

    if (world_rank == world_size - 1) {
        U(grid_size[world_rank]) = U(grid_size[world_rank]-2);
    }
// END: T6
}


// TASK: T5
// Communicate the border between processes.
void border_exchange( void )
{
// BEGIN: T5
    if (world_rank > 0) {
        // send my leftmost point, if I am not the leftmost process
        MPI_Send(&U(0), 1, MPI_DOUBLE, world_rank - 1, 0, MPI_COMM_WORLD);
    }
    if (world_rank < world_size - 1) {
        // receive my right ghost point, if I am not the rightmost process
        MPI_Recv(&U(grid_size[world_rank]), 1, MPI_DOUBLE, world_rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // send my right most point
        MPI_Send(&U(grid_size[world_rank] - 1), 1, MPI_DOUBLE, world_rank + 1, 0, MPI_COMM_WORLD);
    }

    if (world_rank > 0) {
        // receive my left ghost point, if I am not the leftmost process
        MPI_Recv(&U(-1), 1, MPI_DOUBLE, world_rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
// END: T5
}


// TASK: T7
// Every process needs to communicate its results
// to root and assemble it in the root buffer
void send_data_to_root()
{
// BEGIN: T7
    if (world_rank > 0) {
        // send my data to root
        MPI_Send(&U(0), grid_size[world_rank], MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    } else {
        // receive from all others
        int buffer_index = 1 + grid_size[0];
        for (int i = 1; i < world_size; ++i) {
            MPI_Recv(&buffers[1][buffer_index], grid_size[i], MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            buffer_index += grid_size[i];
        }
    }

// END: T7
}


// Main time integration.
void simulate( void )
{
    // Go through each time step.
    for ( int_t iteration=0; iteration<=max_iteration; iteration++ )
    {
        if ( (iteration % snapshot_freq)==0 )
        {
            send_data_to_root();
            domain_save ( iteration / snapshot_freq );
        }

        // Derive step t+1 from steps t and t-1.
        border_exchange();
        boundary_condition();
        time_step();

        move_buffer_window();
    }
}


int main ( int argc, char **argv )
{
// TASK: T1c
// Initialise MPI
// BEGIN: T1c
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
// END: T1c
    
    struct timeval t_start, t_end;

    domain_initialize();

// TASK: T2
// Time your code
// BEGIN: T2
    if (world_rank == 0) {
        gettimeofday( &t_start, NULL );
    }

    simulate();


    if (world_rank == 0) {
        gettimeofday( &t_end, NULL );
    }

    if (world_rank == 0) {
        printf( "Total elapsed time: %lf seconds\n",
            WALLTIME(t_end) - WALLTIME(t_start)
        );
    }
// END: T2
   
    domain_finalize();

// TASK: T1d
// Finalise MPI
// BEGIN: T1d
    MPI_Finalize();
// END: T1d

    exit ( EXIT_SUCCESS );
}
