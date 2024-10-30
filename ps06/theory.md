# Theory questions

## What is the difference between a function declared as `__global__` and a function declared as `__device__` in CUDA?

A function declared as `__device__` in CUDA is to be called from device kernels only. 
It is used to define 'helper' kernels to modularize computations on the device.

A function declared as `__global__` is a device kernel that can be launched from the host.

## What are some advantages of parallelising with CUDA as opposed to MPI?

Parallelising with CUDA has the advantage of using a shared memory model, which means that in general we don't need explicit
communication of data.

CUDA also has the advantage of running on hardware that is specifically designed for massively parallel computation,
whereas MPI is a more general communication framework for distributing work across multiple processors (even over the wire).

## What are some pros and limitations when using cooperative groups?

## How good occupancy did you achieve on the GPU?
