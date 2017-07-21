#include <stdio.h>
#include <stdlib.h>
#include <sched.h>
#include <omp.h>

// Escape time algorithm
// https://en.wikipedia.org/wiki/Mandelbrot_set
#pragma acc routine seq
unsigned int CalculatePixel(double xO, double yO, double pixel_size, const int max_iter) {

  const double escape_radius = 2.0;
  double x = 0.0;
  double y = 0.0;
  int dwell = 0;

  while( (x*x + y*y) < escape_radius*escape_radius && dwell < max_iter) {

    const double tmp_x = x*x - y*y + xO;
    y = 2.0*x*y + yO;
    x = tmp_x;
    dwell++;

  }

	return dwell;

}

int main(int argc, char **argv) {

  // Image bounds
  const double center_x = -0.75;
  const double center_y =  0.0;
  const double length_x =  2.75;
  const double length_y =  2.0;

  // Convenience variables based on image bounds
  const double x_min = center_x - length_x/2.0;
  const double y_min = center_y - length_y/2.0;
  const double pixel_size = 0.0001;
  const int pixels_x = length_x / pixel_size;
  const int pixels_y = length_y / pixel_size; 

	unsigned int iteration;
	const int max_iterations = 50;

	// Define number of blocks/tiles and calcualte block height
	unsigned int num_blocks = 16;
	unsigned int block_height = pixels_y / num_blocks;

  // Linearized 2D image data
  size_t pixel_bytes = sizeof(unsigned char)*pixels_x*pixels_y;
  unsigned char * pixels = (unsigned char*)malloc(pixel_bytes);

	// Determine which physical CPU core is being used
  int cpu = sched_getcpu();
  printf("cpu %d\n", cpu);

	double start_time = omp_get_wtime();

#pragma acc data create(pixels[0:pixels_x*pixels_y])
{
	for(int block=0; block<num_blocks; block++){

		unsigned int y_start = block*block_height;
		unsigned int y_stop  = y_start + block_height;
	
		// Iterate over each pixel and calculate color
		#pragma acc parallel loop async(block % 3 + 1)
	  for(int n_y=y_start; n_y<y_stop; n_y++) {
  	  for(int n_x=0; n_x<pixels_x; n_x++) {

    	  double x = x_min + n_x * pixel_size;
				double y = y_min + n_y * pixel_size;

				iteration = CalculatePixel(x, y, pixel_size, max_iterations);

				// Calculate color based upon escape iterations
	  		if(iteration >= max_iterations) {
    			pixels[n_y * pixels_x + n_x] = 0;
	  		}
				else {
    			pixels[n_y * pixels_x + n_x] = 255*iteration/max_iterations;
  			}

	    }
	  }
		#pragma acc update self(pixels[block*block_height*pixels_x:block_height*pixels_x]) async(block % 3 + 1)

	}

}
#pragma acc wait

  double stop_time = omp_get_wtime();
  double elapsed_time = stop_time - start_time;
  printf("Elapsed Time (s): %f\n", elapsed_time);

  // Write pixels to PGM P5 formatted file
  FILE *file = fopen("mandelbrot.pgm", "wb");
  fprintf(file, "P5\n%d %d\n%d\n", pixels_x, pixels_y, 255);
  fwrite(pixels, sizeof(unsigned char), pixels_x*pixels_y, file);
	fclose(file);

  free(pixels);

  return 0;
}
