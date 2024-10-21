# Theory questions
## Did you get a speedup? Why or why not?

Timings of a typical run my machine:

```
Host time:            2949.839 ms
Device calculation:   0.214 ms
Copy result:          16.742 ms
```

When excluding the Device -> Host transfer time, the speedup is 
$$
\frac{2949.839}{0.214} \approx 13784
$$

Including the transfer time we get 
$$
\frac{2949.839}{16.956} \approx 174
$$

In both cases we get a (big) speedup, but notice the bottle neck of memory transfer. This is often the case with programming on GPUs.

The speedup is expected, as we effectively assign each pixel to each own thread, so we are limited by the number of cores available.

## Which GPU did you use?

Output of `nvidia-smi`:

```
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.183.06             Driver Version: 535.183.06   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 3090 Ti     On  | 00000000:2B:00.0  On |                  Off |
|  0%   53C    P8              27W / 450W |    542MiB / 24564MiB |     19%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
```

The Compute Capability is 8.6.

## Explain the difference between SIMD and SPMD. Which one is CUDA?

SIMD stands for Single Instruction Multiple Data. In this model we think of one instruction unit operating on several streams of data.
This means that the exact same instruction is applied to a 'vector' of data at once.

SPMD, Single Program Multiple Data on the other hand is writing a program that is supposed to run in different processes. 
For each process, branching will occur based on the rank of that process. The data can be shared or distributed, but either way the global data is 
usually split in some way between the ranks.

Technically (according to Nvidias own terms), CUDA is SIMT, Single Instruction Multiple Thread. This can however be thought of as launching several 
SIMD processors in parallel (threads). The distinction between SIMD and SIMT, is that although the threads on a Streaming Multiprocessor execute the same stream of 
instructions, they may not do so simultaneously.

One could argue that writing CUDA programs is also some kind of SPMD, as we usually write the Device (GPU) code alongside the Host (CPU) code in the same program, thus writing
two programs at once.
