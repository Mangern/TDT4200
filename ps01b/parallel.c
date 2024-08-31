#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define XSIZE 2560
#define YSIZE 2048

#define MAXITER 255

typedef double real_t;
typedef unsigned char uchar;

real_t xleft=-2.01;
real_t xright=1;
real_t yupper,ylower;
real_t ycenter=1e-6;
real_t step;


#define COLOR_PIXEL(i,j) ((i)*3 + (j)*XSIZE*3)

typedef struct {
	double real,imag;
} complex_t;


int world_size;
int rank;

uchar* color_chunk;
int *row_chunks;

int my_row_start;
int my_row_end;

void setup();
void calculate();
void cleanup();
void savebmp(char*, uchar*, int, int);
void fancycolour(uchar*, int);


int main(void) {
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    setup();

    calculate();

    if (rank == 0) {
        int row_counter = row_chunks[0];
		unsigned char *buffer=calloc(XSIZE*YSIZE*3,1);
        memcpy(buffer, color_chunk, row_counter * XSIZE * 3);

        for (int r = 1; r < world_size; ++r) {
            MPI_Recv(&buffer[COLOR_PIXEL(0, row_counter)], (row_chunks[r]) * XSIZE * 3, MPI_UNSIGNED_CHAR, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            row_counter += row_chunks[r];
        }

		savebmp("mandel_parallel.bmp",buffer,XSIZE,YSIZE);
    } else {
        MPI_Send(color_chunk, row_chunks[rank] * XSIZE * 3, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD);
    }

    cleanup();

    // Finalize the MPI environment.
    MPI_Finalize();
}

void setup() {
    row_chunks = malloc(world_size * sizeof(int));
    for (int r = 0; r < world_size; ++r)
        row_chunks[r] = (int)( YSIZE / world_size ) + ((r < (YSIZE % world_size)) ? 1 : 0);

    for (int r = 0; r < rank; ++r)
        my_row_start += row_chunks[r];

    my_row_end = my_row_start + row_chunks[rank];

	/* Calculate the range in the y-axis such that we preserve the
	   aspect ratio */
	step=(xright-xleft)/XSIZE;
	yupper=ycenter+(step*YSIZE)/2;
	ylower=ycenter-(step*YSIZE)/2;

    color_chunk = malloc( row_chunks[rank] * XSIZE * 3 );
}

void calculate() {
    for(int j = my_row_start; j < my_row_end; j++) {
	    for(int i = 0; i < XSIZE; i++) {
			/* Calculate the number of iterations until divergence for each pixel.
			   If divergence never happens, return MAXITER */
			complex_t c,z,temp;
			int iter=0;
			c.real = (xleft + step*i);
			c.imag = (ylower + step*(YSIZE - 1 - j)); // invert here! so we don't need to do it later
			z = c;
			while(z.real*z.real + z.imag*z.imag < 4) {
				temp.real = z.real*z.real - z.imag*z.imag + c.real;
				temp.imag = 2*z.real*z.imag + c.imag;
				z = temp;
				if(++iter==MAXITER) break;
			}

            fancycolour(&color_chunk[COLOR_PIXEL(i, j - my_row_start)], iter);
		}
	}
}

void cleanup() {
    free(row_chunks);

    free(color_chunk);
}


/* save 24-bits bmp file, buffer must be in bmp format: upside-down */
void savebmp(char *name,uchar *buffer,int x,int y) {
	FILE *f=fopen(name,"wb");
	if(!f) {
		printf("Error writing image to disk.\n");
		return;
	}
	unsigned int size=x*y*3+54;
	uchar header[54]={'B','M',size&255,(size>>8)&255,(size>>16)&255,size>>24,0,
		0,0,0,54,0,0,0,40,0,0,0,x&255,x>>8,0,0,y&255,y>>8,0,0,1,0,24,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
	fwrite(header,1,54,f);
	fwrite(buffer,1,XSIZE*YSIZE*3,f);
	fclose(f);
}

/* given iteration number, set a colour */
void fancycolour(uchar *p,int iter) {
	if(iter==MAXITER);
	else if(iter<8) { p[0]=128+iter*16; p[1]=p[2]=0; }
	else if(iter<24) { p[0]=255; p[1]=p[2]=(iter-8)*16; }
	else if(iter<160) { p[0]=p[1]=255-(iter-24)*2; p[2]=255; }
	else { p[0]=p[1]=(iter-160)*2; p[2]=255-(iter-160)*2; }
}

