# What is the purpose of a `Makefile`?
The purpose of a Makefile is to automate the compilation process. 
In bigger projects, you typically have several options and compilation flags that you want to use, and possibly several steps in the
compilation of the project. A Makefile lets you organize all the steps needed to compile the project in one place. It also 
lets you declare dependencies between the compilation steps, ensuring that all necessary steps are done when a rule is make-d.
Make will automatically determine which parts of the project needs to be recompiled after a change has been made.

# What is a pointer in C?
A pointer is a variable holding the address of a memory location. 
The pointer also has a type, indicating how the data at the memory location is interpreted. Array indexing makes it possible
to access data at a certain offset from the pointers memory address.

# What does the -O3 flag do when compiling C code?
The -O3 turns on optimization when compiling the C code. 
There are several levels, -O2 and -O3 are both aimed at making the executable faster. It may become harder to debug, as
the produced machine code you get in the debugger is different from the actual source code.
They are actually just shorthands for several optimization flags at once.
From the man page:
```txt
-O3 Optimize  yet  more.   -O3 turns on all optimizations specified by -O2
    and also turns on the following optimization flags:

    -fgcse-after-reload         -fipa-cp-clone          -floop-interchange
    -floop-unroll-and-jam        -fpeel-loops       -fpredictive-commoning
    -fsplit-loops          -fsplit-paths          -ftree-loop-distribution
    -ftree-partial-pre      -funswitch-loops     -fvect-cost-model=dynamic
    -fversion-loops-for-strides
```

# How can you pass a value by reference to a function in C?
To pass a value by reference in C you have to pass a pointer to the value. This is different from C++ where references are a
builtin feature, however under the hood C++ references are just pointers.
