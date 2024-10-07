# Theory Questions

# 1. Discuss how using MPI Allgather versus MPI Gather will impact performance of a program.

MPI_Allgather and MPI Gather have different use cases. While MPI_Gather is a many-to-one operation,
MPI_Allgather is a many-to-many operation. MPI_Allgather is therefore a slower operation in general.

However, in order to achieve the same task with MPI_Gather, one would typically have to follow the MPI_Gather 
operation with a broadcast. In most MPI implementations, broadcast and gather communications follow a tree-like pattern.
So an MPI_Gather followed by a MPI_Bcast would most likely look like two tree-structured communication patterns.

MPI_Allgather on the other hands usually follow a butterfly-pattern instead. This has a lower communication overhead when 
doing a many-to-many operation.

# 2. Let’s say we wanted to use a 9-point stencil for approximation on the current time step instead of just 5, as illustrated in Figure 11. How would you communicate the value needed from Rank 2?

There are several ways to do this.

One way is for the two processes to calculate each other's ranks by for example using MPI_Cart_rank. 
MPI_Cart_rank will translate Cartesian process coordinates to the process ranks.
The results could then be communicated as normal by using MPI_Send/Recv (or Isend/IRecv).

Another way could be to set up a scheme where the processes in between transmit the data. 
So in the example, Rank 2 could send the corner point to Rank 0, which then could send it to Rank 1. This seems like a more cumbersome and slower approach.

Yet another way is to manually set up a graph communicator with the desired structure. Setting up the communicator would be more work, but this may be easier to generalize
to other patterns.
