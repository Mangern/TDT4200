#include <stdlib.h>
#include "bitmap.h"

#define XSIZE 2560 // Size of before image
#define YSIZE 2048

void swap(uchar* a, uchar* b) {
    // https://en.wikipedia.org/wiki/XOR_swap_algorithm
    *a ^= *b;
    *b ^= *a;
    *a ^= *b;
}

/*
 * Flips the image around the vertical axis
 */
void flip_x(uchar* image, int width, int height) {
    for (int i = 0; i < height; ++i) {
        // j here refers to "pixel at column j"
        for (int j = 0; j < (width >> 1); ++j) {
            // c is the channel offset
            for (int c = 0; c < 3; ++c) {
                int src = i * width * 3 + j * 3 + c;
                int dst = i * width * 3 + (width - j - 1) * 3 + c;
                swap(image+src, image+dst);
            }
        }
    }
}

/*
 * Flips the image around the horizontal axis
 */
void flip_y(uchar* image, int width, int height) {
    for (int i = 0; i < (height >> 1); ++i) {
        // j here refers to byte at index j
        for (int j = 0; j < width * 3; ++j) {
            int src = i * width * 3 + j;
            int dst = (height - 1 - i) * width * 3 + j;
            swap(image+src, image+dst);
        }
    }
}

/*
 * Utility method for copying data between two images of different sizes.
 * Uses stupid interpolation. 
 */
void copy_resize(
    uchar* dst, 
    int dst_width, 
    int dst_height, 
    uchar* src, 
    int src_width, 
    int src_height) {
    for (int i = 0; i < dst_height; ++i) {
        for (int j = 0; j < dst_width; ++j) {
            for (int c = 0; c < 3; ++c) {
                int dst_idx = i * dst_width * 3 + j * 3 + c;

                // This will essentially duplicate pixels 
                // when dst is bigger than src
                int src_i = (i * src_height) / dst_height;
                int src_j = (j * src_width) / dst_width;

                int src_idx = src_i * src_width * 3 + src_j * 3 + c;
                dst[dst_idx] = src[src_idx];
            }
        }
    }
}

/*
 * Creates a new image with double the size of the passed image.
 * Will free the old image pointer and replace it with a pointer to the resized image.
 * So make sure no other pointers to the original image data is used.
 */
void double_size(uchar** image, int width, int height) {
    uchar* resized = calloc(width * height * 3 * 4, 1);
    copy_resize(resized, width * 2, height * 2, *image, width, height);

    uchar* old_image = *image;
    *image = resized;

    free(old_image);
}

/*
 * Inverts the first channel of the image for all pixels 
 */
void invert_r_channel(uchar* image, int width, int height) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            // 255 is max value for uchar
            image[i * width * 3 + j * 3] = 255 - image[i * width * 3 + j * 3];
        }
    }
}

int main(void)
{
	uchar *image = calloc(XSIZE * YSIZE * 3, 1); // Three uchars per pixel (RGB)
	readbmp("before.bmp", image);

	// Alter the image here
	flip_x(image, XSIZE, YSIZE);
	flip_y(image, XSIZE, YSIZE);

    // Invert before resize because it is faster
    invert_r_channel(image, XSIZE, YSIZE);

    double_size(&image, XSIZE, YSIZE);

	savebmp("after.bmp", image, XSIZE * 2, YSIZE * 2);

    free(image);
	return 0;
}
