# 2.2

## Question 1 
*What speed-up / expected speed-up did you get by parallelising the code?*

## Question 2
*What is the name for this type of parallelisation (splitting the domain between processes and using ghost cells to communicate)?*

## Question 3 
*What is the difference between point-to-point communication and collective communication?*

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
```
