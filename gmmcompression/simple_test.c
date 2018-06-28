#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "mixture.h"

int main(void) {
    
    printf("Loading mixture\n");
    
    FILE* in = fopen( "testdata/test_mixture.dat", "r" );
    Mixture* m = mixture_load( in );
    fclose( in );
    
    printf("Loading grid\n");
    
    //HACK
    int nx = 27;
    in = fopen( "testdata/xgrid.dat", "r" );
    double* xgrid = malloc( nx*sizeof(double) );
    fread( xgrid, sizeof(double), nx, in );
    fclose( in );
    
    printf("Loading test points\n");
    
    int ny = 26;
    in = fopen( "testdata/ygrid.dat", "r" );
    double* ygrid = malloc( ny*sizeof(double) );
    fread( ygrid, sizeof(double), ny, in );
    fclose( in );
    
    printf("Constructing output array\n");
    
    double* result = malloc( nx*ny*sizeof(double) );
    uint16_t griddim = 0;
    uint16_t pointdim = 1;
    
    struct timespec ts_start;
    struct timespec ts_end;
    
    printf("Updating cache\n");
    mixture_update_cache( m );
    
    double* grid_acc = NULL;
	grid_acc = malloc( m->ncomponents * nx * sizeof(double*) );
	
	mixture_prepare_grid_accumulator( m, xgrid, nx, 1, &griddim, grid_acc );
    
    printf("Getting ready to evaluate\n" );
    
	clock_gettime(CLOCK_MONOTONIC, &ts_start);
    
    int loop;
    int nloop = 1000;
    for (loop=0;loop<nloop;loop++) {
        mixture_evaluategrid_diagonal_multi( m, grid_acc, nx, ygrid, ny, 1, &pointdim, result ); 
    }

    clock_gettime(CLOCK_MONOTONIC, &ts_end);

    double elapsed_time = (double) (ts_end.tv_sec-ts_start.tv_sec)*1000.0 + (double) (ts_end.tv_nsec-ts_start.tv_nsec)/1000000.0;
    printf( "total time: %f ms (nloops = %d, ngrid = %d, npoints = %d)\n", elapsed_time, nloop, nx, ny );
    printf( "time per spike: %f ms\n", elapsed_time / (nloop * ny) );
    
    
    printf( "Saving result to result.dat\n");
    
    FILE* out = fopen( "testdata/result.dat", "w" );
    fwrite( result, sizeof(double), nx*ny, out );
    fclose(out);
    
    free(result);
    free(ygrid);
    free(xgrid);
    
    return 0;
}
