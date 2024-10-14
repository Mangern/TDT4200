# Theory
## 1. Why is there no need for a border exchange when using Pthreads?
When using Pthreads we use a shared memory model. All threads have access to the global state. 
Therefore it is unnecessary to include communicating the border points between threads.

## 2. What is the difference between OpenMP and MPI?
MPI is a communication interface for distributed memory parallel programming, while OpenMP 
is a standard for implementing shared-memory models.

## 3. Comment on the difference between Pthreads and the two OpenMP implementations.
The Pthreads implementation requires the most manual handling of threads.

This is no surprise considering OpenMP uses pthreads under the hood to do the actual parallelisation.

The `wave_2d_barrier.c` bears the most similarities to the Pthreads implementation, as constructs such as

```c
int_t thread_id = omp_get_thread_num();
```

and 

```c
#pragma omp barrier
```

in OpenMP are analog to 

```c
// Pass the thread id to the simulate function
pthread_create(&thread_handles[i], NULL, &simulate, (void*)i);
```

and 

```c
pthread_barrier_wait(&barrier_stop);
```

in Pthreads. However, OpenMP still abstracts away some of the thread creation and joining.

The "Workshare" OpenMP implementation on the other hand bears fewer similarities to the Pthreads implementation.
In that version most of the actual 'parallelisation' is abstracted away during the call:

```c
#pragma omp parallel for
```

This makes the code easier to reason about, and lets OpenMP do the heavy lifting.

On my computer I got the best performance from Pthreads, acheiving 8.28 seconds with 13 threads, while 
the OpenMP workshare version used 12.33 seconds for the same task with 16 threads.

I suspect it may have something to do with how I distribute the load as well, as in my Pthreads implementation 
i give each thread a contiguous set of rows, while the OpenMP version may decide to scatter the rows among the threads.
As far as I can tell, each core on my computer has access to a 32kB L1 cache, which may fit a few rows of the grid at once.

## 4. How would you parallelise a recursion problem with OpenMP?

There could be a couple ways to go about this.

Every recursive algorithm can be implemented iteratively and vice versa. 
In some instances this might be the best way to achieve parallelism, especially when the iterative implementation
consists of tight for-loops.

In other cases however, the iterative equivalent has to rely on a stack datastructure to 'simulate' the recursive execution of the algorithm,
which could require exessive use of barriers to synchronize the stack.

Another way could be to use OpenMP `task` directive, for example:

```c
int my_recursive_computation(int arg) {
    int res = 0;
    #pragma omp taskgroup task_reduction(+: res)
    {
        #pragma omp task in_reduction(+: res)
        res += my_recursive_computation(...)
        
        #pragma omp task in_reduction(+: res)
        res += my_recursive_computation(...)

        ...
    }

    return res;
}
```

If this wrapped in a `#pragma omp parallel` outside of the function it 
will not exhaust resources as far as I have understood. It may however be harder to utilize the resources properly
doing it this way.
