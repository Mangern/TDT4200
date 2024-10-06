#include <assert.h>
#define _XOPEN_SOURCE 600
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <sys/time.h>

#include "argument_utils.h"

// TASK: T1a
// Include the MPI headerfile
// BEGIN: T1a
#include <mpi.h>
// END: T1a


// Convert 'struct timeval' into seconds in double prec. floating point
#define WALLTIME(t) ((double)(t).tv_sec + 1e-6 * (double)(t).tv_usec)

// Option to change numerical precision
typedef int64_t int_t;
typedef double real_t;


// Buffers for three time steps, indexed with 2 ghost points for the boundary
real_t
    *buffers[3] = { NULL, NULL, NULL };

// TASK: T1b
// Declare variables each MPI process will need
// BEGIN: T1b
#define MPI_RANK_ROOT ( world_rank == 0 )
#define MPI_RANK_LAST ( world_rank == world_size - 1 )
#define NUM_COMM_DIMS 2

// Redefine the macros to subtract the offset we have in our rank
#define U_prv(i,j) buffers[0][(((i)-subgrid_loc[0])+1)*(subgrid_dims[1]+2)+((j)-subgrid_loc[1])+1]
#define U(i,j)     buffers[1][(((i)-subgrid_loc[0])+1)*(subgrid_dims[1]+2)+((j)-subgrid_loc[1])+1]
#define U_nxt(i,j) buffers[2][(((i)-subgrid_loc[0])+1)*(subgrid_dims[1]+2)+((j)-subgrid_loc[1])+1]

int world_size;
int world_rank;

MPI_Comm comm_cart;
int comm_dims[2] = { 0, 0 };
int comm_coords[2] = { 0, 0 };

int subgrid_dims[2] = { 0, 0 };
int subgrid_loc[2] = { 0, 0 };
int* subgrid_rows_sz;
int* subgrid_cols_sz;

MPI_Request *send_requests;
MPI_Request *recv_requests;
MPI_Datatype column_vector_type;

int neigbors[2][2];
// END: T1b

// Simulation parameters: size, step count, and how often to save the state
// Store explicitly as int64_t because we want to use them as mpi types
int64_t
    M = 256,    // rows
    N = 256,    // cols
    max_iteration = 4000,
    snapshot_freq = 20;

// Wave equation parameters, time step is derived from the space step
const real_t
    c  = 1.0,
    dx = 1.0,
    dy = 1.0;
real_t
    dt;

// Rotate the time step buffers.
void move_buffer_window ( void )
{
    real_t *temp = buffers[0];
    buffers[0] = buffers[1];
    buffers[1] = buffers[2];
    buffers[2] = temp;
}


// TASK: T4
// Set up our three buffers, and fill two with an initial perturbation
// and set the time step.
void domain_initialize ( void )
{
// BEGIN: T4
    subgrid_rows_sz = malloc((comm_dims[0] * sizeof(int)));
    subgrid_cols_sz = malloc((comm_dims[1] * sizeof(int)));

    for (int i = 0; i < comm_dims[0]; ++i) {
        subgrid_rows_sz[i] = M / comm_dims[0] + (i < (M % comm_dims[0]) ? 1 : 0);

        if (i < comm_coords[0]) {
            subgrid_loc[0] += subgrid_rows_sz[i];
        }
    }

    for (int i = 0; i < comm_dims[1]; ++i) {
        subgrid_cols_sz[i] = N / comm_dims[1] + (i < (N % comm_dims[1]) ? 1 : 0);

        if (i < comm_coords[1]) {
            subgrid_loc[1] += subgrid_cols_sz[i];
        }
    }

    subgrid_dims[0] = subgrid_rows_sz[comm_coords[0]];
    subgrid_dims[1] = subgrid_cols_sz[comm_coords[1]];

    // Rows are trivially contiguous, columns have a stride
    MPI_Type_vector(subgrid_dims[0], 1, subgrid_dims[1] + 2, MPI_DOUBLE, &column_vector_type);
    MPI_Type_commit(&column_vector_type);

    // Allocate only our own subgrid
    buffers[0] = malloc ( (subgrid_dims[0]+2)*(subgrid_dims[1]+2)*sizeof(real_t) );
    buffers[1] = malloc ( (subgrid_dims[0]+2)*(subgrid_dims[1]+2)*sizeof(real_t) );
    buffers[2] = malloc ( (subgrid_dims[0]+2)*(subgrid_dims[1]+2)*sizeof(real_t) );

    // Send will happen along all four borders
    // Allocate twice needed for send_requests
    send_requests = malloc ( sizeof(MPI_Request) * 4 * 2 );
    // and let recv_requests point to last half to get a single Waitall for everything.
    recv_requests = &send_requests[4];

    MPI_Cart_shift(comm_cart, 0, 1, &neigbors[0][0], &neigbors[0][1]);
    MPI_Cart_shift(comm_cart, 1, 1, &neigbors[1][0], &neigbors[1][1]);

    if (world_rank == 0) {
        // debug assert
        assert(neigbors[0][0] == MPI_PROC_NULL);
        assert(neigbors[1][0] == MPI_PROC_NULL);
    }

    //printf("%d, (i, j) = (%d, %d), (h, w) = (%d, %d)\n", world_rank, subgrid_loc[0], subgrid_loc[1], subgrid_dims[1], subgrid_dims[1]);

    for ( int_t i=subgrid_loc[0]; i<subgrid_loc[0] + subgrid_dims[0]; i++ )
    {
        for ( int_t j=subgrid_loc[1]; j<subgrid_loc[1] + subgrid_dims[1]; j++ )
        {
            // Calculate delta (radial distance) adjusted for M x N grid
            real_t delta = sqrt ( ((i - M/2.0) * (i - M/2.0)) / (real_t)M +
                                ((j - N/2.0) * (j - N/2.0)) / (real_t)N );
            U_prv(i,j) = U(i,j) = exp ( -4.0*delta*delta );
        }
    }

    // Set the time step for 2D case
    dt = dx*dy / (c * sqrt (dx*dx+dy*dy));
// END: T4
}


// Get rid of all the memory allocations
void domain_finalize ( void )
{
    free ( buffers[0] );
    free ( buffers[1] );
    free ( buffers[2] );

    free(subgrid_rows_sz);
    free(subgrid_cols_sz);

    // also 'frees' recv_requests because its the same buffer
    free(send_requests);
}


// TASK: T5
// Integration formula
void time_step ( void )
{
// BEGIN: T5
    for ( int_t i=subgrid_loc[0]; i<subgrid_loc[0] + subgrid_dims[0]; i++ )
    {
        for ( int_t j=subgrid_loc[1]; j<subgrid_loc[1] + subgrid_dims[1]; j++ )
        {
            U_nxt(i,j) = -U_prv(i,j) + 2.0*U(i,j)
                     + (dt*dt*c*c)/(dx*dy) * (
                        U(i-1,j)+U(i+1,j)+U(i,j-1)+U(i,j+1)-4.0*U(i,j)
                    );
        }
    }
// END: T5
}

// TASK: T6
// Communicate the border between processes.
void border_exchange ( void )
{
// BEGIN: T6
    // Simultaneous request counter and tag
    int tag = 0;

    // send at top row, receive at bottom
    MPI_Isend(&U(subgrid_loc[0], subgrid_loc[1]),                   subgrid_dims[1], MPI_DOUBLE, neigbors[0][0], tag, comm_cart, &send_requests[tag]);
    MPI_Irecv(&U(subgrid_loc[0] + subgrid_dims[0], subgrid_loc[1]), subgrid_dims[1], MPI_DOUBLE, neigbors[0][1], tag, comm_cart, &recv_requests[tag]);
    ++tag;

    // send bottom row, receive top
    MPI_Isend(&U(subgrid_loc[0] + subgrid_dims[0] - 1, subgrid_loc[1]), subgrid_dims[1], MPI_DOUBLE, neigbors[0][1], tag, comm_cart, &send_requests[tag]);
    MPI_Irecv(&U(subgrid_loc[0] - 1, subgrid_loc[1]),                   subgrid_dims[1], MPI_DOUBLE, neigbors[0][0], tag, comm_cart, &recv_requests[tag]);
    ++tag;

    // send left col, receive right
    MPI_Isend(&U(subgrid_loc[0], subgrid_loc[1]),                   1, column_vector_type, neigbors[1][0], tag, comm_cart, &send_requests[tag]);
    MPI_Irecv(&U(subgrid_loc[0], subgrid_loc[1] + subgrid_dims[1]), 1, column_vector_type, neigbors[1][1], tag, comm_cart, &recv_requests[tag]);
    ++tag;

    // send right col, receive left
    MPI_Isend(&U(subgrid_loc[0], subgrid_loc[1] + subgrid_dims[1] - 1), 1, column_vector_type, neigbors[1][1], tag, comm_cart, &send_requests[tag]);
    MPI_Irecv(&U(subgrid_loc[0], subgrid_loc[1] - 1),                   1, column_vector_type, neigbors[1][0], tag, comm_cart, &recv_requests[tag]);
    ++tag;

    // tag holds number of sends (or receives)
    // double this to wait for all sends and receives
    MPI_Waitall(2*tag, send_requests, MPI_STATUSES_IGNORE);

// END: T6
}


// TASK: T7
// Neumann (reflective) boundary condition
void boundary_condition ( void )
{
// BEGIN: T7
    if (comm_coords[1] == 0) {
        for ( int_t i=subgrid_loc[0]; i<subgrid_loc[0] + subgrid_dims[0]; i++ )
        {
            U(i,-1) = U(i,1);
        }
    }
    if (comm_coords[1] == comm_dims[1] - 1) {
        for ( int_t i=subgrid_loc[0]; i<subgrid_loc[0] + subgrid_dims[0]; i++ )
        {
            U(i,N)  = U(i,N-2);
        }
    }
    if (comm_coords[0] == 0) {
        for ( int_t j=subgrid_loc[1]; j<subgrid_loc[1] + subgrid_dims[1]; j++ )
        {
            U(-1,j) = U(1,j);
        }
    }
    if (comm_coords[0] == comm_dims[0] - 1) {
        for ( int_t j=subgrid_loc[1]; j<subgrid_loc[1] + subgrid_dims[1]; j++ )
        {
            U(M,j)  = U(M-2,j);
        }
    }
// END: T7
}


// TASK: T8
// Save the present time step in a numbered file under 'data/'
void domain_save ( int_t step )
{
// BEGIN: T8
    char filename[256];
    sprintf ( filename, "data/%.5ld.dat", step );
    MPI_File out;
    MPI_File_open(
        comm_cart, 
        filename,
        MPI_MODE_CREATE | MPI_MODE_WRONLY, 
        MPI_INFO_NULL, 
        &out);

    int offs = subgrid_loc[1];

    for (int i = 0; i < subgrid_loc[0]; ++i) {
        offs += N;
    }

    for (int i = subgrid_loc[0]; i < subgrid_loc[0] + subgrid_dims[0]; ++i) {
        MPI_File_write_at(
            out, 
            offs * sizeof(real_t), 
            &U(i, subgrid_loc[1]), 
            subgrid_dims[1], 
            MPI_DOUBLE, 
            MPI_STATUS_IGNORE
        );
        offs += N;
    }

    MPI_File_close(&out);
// END: T8
}


// Main time integration.
void simulate( void )
{
    // Go through each time step
    for ( int_t iteration=0; iteration<=max_iteration; iteration++ )
    {
        if ( (iteration % snapshot_freq)==0 )
        {
            domain_save ( iteration / snapshot_freq );
        }

        // Derive step t+1 from steps t and t-1
        border_exchange();
        boundary_condition();
        time_step();

        // Rotate the time step buffers
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


// TASK: T3
// Distribute the user arguments to all the processes
// BEGIN: T3
        if (MPI_RANK_ROOT) {
            OPTIONS *options = parse_args( argc, argv );
            if ( !options )
            {
                fprintf( stderr, "Argument parsing failed\n" );
                exit( EXIT_FAILURE );
            }

            M = options->M;
            N = options->N;
            max_iteration = options->max_iteration;
            snapshot_freq = options->snapshot_frequency;
        }

        MPI_Bcast(&M,             1, MPI_INT64_T, 0, MPI_COMM_WORLD);
        MPI_Bcast(&N,             1, MPI_INT64_T, 0, MPI_COMM_WORLD);
        MPI_Bcast(&max_iteration, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
        MPI_Bcast(&snapshot_freq, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
        int periods[2] = { 0, 0 };

        MPI_Dims_create(world_size, 2, comm_dims);

        MPI_Cart_create(
            MPI_COMM_WORLD, 
            NUM_COMM_DIMS, 
            comm_dims, 
            periods, 
            0, 
            &comm_cart
        );

        MPI_Cart_coords(comm_cart, world_rank, NUM_COMM_DIMS, comm_coords);
// END: T3

    // Set up the initial state of the domain
    domain_initialize();


    struct timeval t_start, t_end;

// TASK: T2
// Time your code
// BEGIN: T2
    if (MPI_RANK_ROOT) {
        gettimeofday( &t_start, NULL );
    }

    simulate();
    if (MPI_RANK_ROOT) {

        gettimeofday( &t_end, NULL );

        printf( "Total elapsed time: %lf seconds\n",
            WALLTIME(t_end) - WALLTIME(t_start)
        );
    }
// END: T2

    // Clean up and shut down
    domain_finalize();

// TASK: T1d
// Finalise MPI
// BEGIN: T1d
    MPI_Finalize();
// END: T1d

    exit ( EXIT_SUCCESS );
}
