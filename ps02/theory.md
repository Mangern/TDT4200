# 2.2

## Question 1 
*What speed-up / expected speed-up did you get by parallelising the code?*

Here are some runs on my local AMD Ryzen 5 5600X 6-Core Processor:

|         | Sequential | Parallel n=1 | Parallel n=4 | Parallel n=13|
| ----    |:----------:|:----------:  |:----------:  |:----------:  |
| Run 1   | 4.41s      | 4.70s        | 1.29s        | 1.99s        |
| Run 2   | 4.42s      | 4.43s        | 1.34s        | 2.25s        |
| Run 3   | 4.42s      | 4.44s        | 1.27s        | 2.00s        |
| Run 4   | 4.46s      | 4.43s        | 1.27s        | 2.03s        |
| Run 5   | 4.41s      | 4.43s        | 1.28s        | 1.96s        |
| Average | 4.42s      | 4.49s        | 1.29s        | 2.05s        |
| Speedup | 1.00x      | 0.98x        | 3.43x        | 2.15x        |

When `n>6` it is oversubscribed, and we can see a worse performance.

## Question 2
*What is the name for this type of parallelisation (splitting the domain between processes and using ghost cells to communicate)?*

The type of parallelisation we use in this exercise is SPMD, Single Program Multiple Data. 
Because there is only one program, but the processes branch based on their rank. The data is split between them.

## Question 3 
*What is the difference between point-to-point communication and collective communication?*

In MPI, functions that transfer data from one rank to exactly one other rank are called point-to-point communications.

`MPI_Send` and `MPI_Recv` are such functions. They have one source rank calling `MPI_Send` and one destination rank calling `MPI_Recv`.

On the contrary, functions that involve communication among all the ranks in a communicator are collective communications.

Examples are `MPI_Bcast` (broadcast) and `MPI_Scatter`.

There are some key differences between the point-to-point communications and the collective ones:
- When using collective communication, all processes must use the exact same function. 
- The function calls must have compatible arguments. For example, when using `MPI_Bcast`, the source_proc argument must be equal for all ranks.
- The matching of point-to-point communication calls are based on argument types and tags, while collective functions are matched on the order in which they are called.

## Question 4
*Given the following code:*
```c
int* a, b;
```
*Which type is `b`?*

`b` is of type `int`. It would have been more helpful to write:

```c
int *a, b;
```

Because then it is clearer what the order of precedence is. 

To make both be of type `int*`, we could write:

```c
int *a, *b;
``
