EXECS=parallel
MPICC?=mpicc

all: ${EXECS}

parallel: parallel.c
	${MPICC} -o parallel parallel.c

clean:
	rm ${EXECS}
