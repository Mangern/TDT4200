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

# 3. Try to run the code with M = 2048 and N = 512. What changes, and why does this happen? 

The first thing to notice is something wrong with the plotting code.
When running 
```bash
mpiexec -n 4 ./parallel -m 2048 -n 512
```
followed by 
```bash
./plot_image.sh -m 2048 -n 512
```
the first image looks like Figure 1. There are four 'spikes' where there should have been one.

![Plot with grid dimension equal to the program's.](./screenshots/quad_spike.png)

However, swapping M and N in `./plot_image.sh` yields the correct plot as shown in Figure 2.
```bash
./plot_image.sh -m 512 -n 2048
```
![Plot with grid dimensions swapped.](./screenshots/wide_spike.png)

I suspect the reason for this is that gnuplot reads the binary data in a column-major order.

So when you write `array=MxN`, it expects the data to be `N` rows of `M` elements each, 
however, when we write the data to the file, we write `M` rows of `N` elements each.

From the gnuplot manual: http://gnuplot.info/docs_6.0/loc7882.html.

This may depend on the version of gnuplot used.

The next thing to notice is the oval shape of the wave. It is easier to see after a few iterations. See Figure 3.
![Oval plot.](./screenshots/oval.png)

This happens because the radial distance used in the initial condition is scaled by the size of each grid dimension.
This makes the initial gaussian have a non-symmetric covariance matrix when `M!=N`.

# 4. What is the difference between *weak scaling* and *strong scaling*?

Scalability means that we can keep the efficiency of a parallel program when 
increasing the number of processes.

Efficiency is defined as speedup per process, i.e.:

$$
\frac{T_{serial}/T_{parallel}}{p}
$$

In general, we expect the efficiency to decrease when increasing the number of processes
due to the increase in prallel overhead.

If the efficiency stays constant for a fixed problem size regardless of the number of processes $p$, the program is said to be **strongly scalable**.

On the other hand, if we can make the efficiency constant by increasing the program size at the same rate as we increase the number of processes the program is **weakly scalable**.

This is according to Pacheco and Malensek (2021).
