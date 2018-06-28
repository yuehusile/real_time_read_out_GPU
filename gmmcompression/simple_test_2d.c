#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "mixture.h"

int main(void) {
    
    struct timespec ts_start;
    struct timespec ts_end;
    double elapsed_time;
    
    printf("Loading mixture\n");
    
    FILE* in = fopen( "testdata/test_tt1_mixture2.dat", "r" );
    Mixture* m = mixture_load( in );
    fclose( in );
    
    printf("Number of components in mixture = %d\n", m->ncomponents );
    
    printf("Loading grid\n");
    //HACK
    int nx = 27;
    int ny = 26;
    int ngriddim = 2;
    in = fopen( "testdata/2dgrid.dat", "r" );
    double* grid = malloc( nx*ny*ngriddim*sizeof(double) );
    fread( grid, sizeof(double), nx*ny*ngriddim, in );
    fclose( in );
    
    printf("First grid point: x=%f, y=%f\n", grid[0], grid[1] );
    
    printf("Loading test spikes\n");
    int nspikes = 106;
    int nspikedim = 4;
    in = fopen( "testdata/tt1_test_spikes.dat", "r" );
    double* spikes = malloc( nspikes*nspikedim*sizeof(double) );
    fread( spikes, sizeof(double), nspikes*nspikedim, in );
    fclose(in);
    
    printf("First spike: %f, %f, %f, %f\n", spikes[0], spikes[1], spikes[2], spikes[3] );
    
    printf("Constructing output array\n");
    double* result = malloc( nx*ny*nspikes*sizeof(double) );
    uint16_t griddim[2] = {0,1};
    uint16_t pointdim[4] = {2,3,4,5};
    

    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    
    printf("Updating cache\n");
    mixture_update_cache( m );
    
    double* grid_acc = NULL;
	uint32_t ngrid = nx*ny;
	grid_acc = malloc( m->ncomponents * ngrid * sizeof(double*) );
	
	mixture_prepare_grid_accumulator( m, grid, ngrid, ngriddim, griddim, grid_acc );
    
    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    
    elapsed_time = (double) (ts_end.tv_sec-ts_start.tv_sec)*1000.0 + (double) (ts_end.tv_nsec-ts_start.tv_nsec)/1000000.0;
    printf("Updating cache took %f ms\n", elapsed_time );
    
    printf("Getting ready to evaluate\n" );
	clock_gettime(CLOCK_MONOTONIC, &ts_start);
    
    int loop;
    int nloop = 10;
    for (loop=0;loop<nloop;loop++) {

        mixture_evaluategrid_diagonal_multi( m, grid_acc, ngrid, spikes, nspikes, nspikedim, pointdim, result ); 
    }

    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    
    elapsed_time = (double) (ts_end.tv_sec-ts_start.tv_sec)*1000.0 + (double) (ts_end.tv_nsec-ts_start.tv_nsec)/1000000.0;
    printf( "total time: %f ms (nloops = %d, ngrid = %d, nspikes = %d)\n", elapsed_time, nloop, nx*ny, nspikes );
    printf( "time per spike: %f ms\n", elapsed_time / (nloop * nspikes) );
    
    printf( "Saving result to result2d_tt1.dat\n" );
    FILE* out = fopen( "testdata/result2d_tt1.dat", "w" );
    fwrite( result, sizeof(double), nx*ny*nspikes, out );
    fclose(out);
    
    free(result);
    free(spikes);
    free(grid);
    
    return 0;
}
