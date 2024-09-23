# 2.2 Theory questions

## Task 1
*Explain the pros and cons of using macros like*

```C
#define U_prv(i) buffers[0][(i)+1]
#define U(i) buffers[1][(i)+1]
#define U_nxt(i) buffers[2][(i)+1]
```

*for simulations like the wave equation.*

Using macros like this abstracts away the underlying buffers when implementing the equations in code. For instance, we don't need to remember that we have space for a "ghost point" in the beginning of each buffer.

It also makes the code easier to relate to the equations, because for example `U(i)` corresponds to the variable $u_i^t$ from the equations.

However, there are also some downsides with using this approach:

- `U(i)` looks like a function call, when it in fact is a substitution of a buffer element
- Macros are inherently hard to debug, as substitution happens in the preprocessing step before the actual compilation. Errors in the macro definitions themselves are particularly difficult to find.
- Abstracting away the underlying buffers takes us further from the actual execution logic, which may result in the code being harder to reason about at a low level.

## Task 2
*Mention at least one other boundary condition that could have been applied instead of the
Neumann (reflective) boundary condition.*

Another type of boundary condition that could be used is Dirichlet boundary condition. This boundary condition states that the function is fixed at the boundary points, i.e. $u(0, t) = u(N, t) = C$. 

For the wave equation, we can imagine this as a string being held tight at two points and oscillating in between.

This is in contrast to the reflective Neumann condition, which is an implementation of the condition that the *derivative* of the function should be constant ($0$) at the boundary points.

## Task 3
*What happens if you don’t allocate memory in T1?*

If you don't allocate memory in T1 you will later be dereferencing null pointers, because the buffer pointers are global variables, meaning that they are initialized to zero. Dereferencing null pointers is undefined behavior in C. If you're lucky you get a segmentation fault.

## Task 4
*What is the difference between*
```c
float const *a;
```
*and*
```c
float* const b;
```
*?*

In this example, `a` and `b` have different types, but both are pointers to float. For `a`, the pointer itself is not `const`, but the data it points to is. So writing
```c
float x = 2.f;
a = &x;
```
is valid, however
```c
*a = 2.f;
```
is not.
For `b` it is the opposite. Here, the pointer is `const`, but not the data. So writing
```c
*b = 3.f;
```
is valid, but
```c
b = &x;
```
is not.

It is even possible to combine them:
```c
float const * const c = &x;
```

to make an unmodifiable pointer to unmodifiable data (for no reason whatsoever).
