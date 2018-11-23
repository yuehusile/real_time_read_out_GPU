#include <stdlib.h>
#include <math.h>
#include <inttypes.h>

//C function for weighted mixed kernel density evaluation
void evaluate_mixed_kde_c( double *data, double *weights, double *x, int32_t *kernels, double *bandwidths, double **distance, int32_t* nlut, int32_t ndim, int32_t npoints, int32_t nx, double cutoff_gauss, double *output ) {

int_fast32_t n,g,q;
double acc_gauss, acc_epa, acc_vonmises, acc_box;
int32_t skip;
double dist;
int32_t ix, idata;

//square gaussian kernel cut off
cutoff_gauss = cutoff_gauss*cutoff_gauss;

for (n=0; n<npoints; n++) { // loop through all data points

    for (g=0; g<nx; g++) { // loop through all evaluation points 
        
        acc_gauss = acc_epa = acc_vonmises = acc_box = 0; //initialize accumulators
        skip = 0;
        
        for ( q=0; q<ndim; q++ ) { // loop through all dimensions
            
            ix = q*nx + g; //TODO: optimize
            idata = q*npoints + n; //TODO: optimize
            
            if (nlut[q]>0) {
                // look-up distance in table
                dist = distance[q][ (int32_t) x[ix] + nlut[q] * (int32_t) data[idata] ];
            } else {
                // compute distance
                dist = x[ix] - data[idata];
            }
            
            if (kernels[q]!=2) {
                dist /= bandwidths[q]; //TODO optimize
            }
            
            if (kernels[q]==0) { // gaussian kernel
                acc_gauss += dist*dist;
                if (acc_gauss>cutoff_gauss) {
                    skip=1;
                    break;
                }
            } else if (kernels[q]==1) { // epanechnikov kernel
                acc_epa += dist*dist;
                if (acc_epa>1) {
                    skip=1;
                    break;
                }
            } else if (kernels[q]==2) { // von mises kernel
                acc_vonmises += bandwidths[q] * cos(dist);
                //TODO: truncated von mises kernel
            } else if (kernels[q]==3) { // box kernel
                acc_box += dist*dist;
                if (acc_box>1) {
                    skip=1;
                    break;
                }
            } else if (kernels[q]==4) { // kronecker delta kernel
                if ( (int32_t) dist > 0 ) {
                    skip=1;
                    break;
                }
            }
        }
        
        if (skip==0) {
            // accumulate contribution of data points
            output[g] += weights[n] * exp( -0.5*acc_gauss + acc_vonmises ) * (1-acc_epa);
        }
        
    }

}

}

// C function for weighted mixed kernel density estimation over a fixed grid
void evaluate_mixed_kde_grid_c( double *data, double *weights, double *x, double *y, int32_t *kernels, double *bandwidths, double **distance, int32_t* nlut, int32_t ndimx, int32_t ndimy, int32_t npoints, int32_t nx, int32_t ny, double cutoff_gauss, double *output ) {

int_fast32_t n,g,q,m,d,d2;
double *acc_gauss, *acc_epa, *acc_vonmises, *acc_box;
double acc_gauss_m, acc_epa_m, acc_vonmises_m, acc_box_m;
int32_t *skip;
double dist;
int32_t ix, idata;

//square gaussian kernel cut off
cutoff_gauss = cutoff_gauss*cutoff_gauss;

// allocate accumulators
skip = (int32_t*) calloc(nx, sizeof(int32_t));
acc_gauss = (double*) calloc(nx, sizeof(double));
acc_epa = (double*) calloc(nx, sizeof(double));
acc_vonmises = (double*) calloc(nx, sizeof(double));
acc_box = (double*) calloc(nx, sizeof(double));


for (n=0; n<npoints; n++) { // loop through all data points

    for (g=0; g<nx; g++) { // loop through all grid points
        
        // initialize accumulators
        acc_gauss[g] = acc_epa[g] = acc_vonmises[g] = acc_box[g] = 0;
        skip[g] = 0;
        
        for ( q=0; q<ndimx; q++ ) { // loop through grid dimensions
            
            ix = q*nx + g; //TODO: optimize (get rid of multiplication)
            idata = q*npoints + n; //TODO: optimize (get rid of multiplication)
            
            if (nlut[q]>0) {
                // look up distance in table
                dist = distance[q][ (int32_t) x[ix] + nlut[q] * (int32_t) data[idata] ];
            } else {
                // compute distance
                dist = x[ix] - data[idata];
            }
            
            if (kernels[q]!=2) {
                dist /= bandwidths[q]; // TODO: optimize (by normalizing data to bandwidth ahead of time)
            }
            
            if (kernels[q]==0) { // gaussian kernel
                acc_gauss[g] += dist*dist;
                if (acc_gauss[g]>cutoff_gauss) {
                    skip[g]=1;
                    break;
                }
            } else if (kernels[q]==1) { // epanechnikov kernel
                acc_epa[g] += dist*dist;
                if (acc_epa[g]>1) {
                    skip[g]=1;
                    break;
                }
            } else if (kernels[q]==2) { // von mises kernel
                acc_vonmises[g] += bandwidths[q] * cos(dist);
                //TODO: truncated von mises kernel
            } else if (kernels[q]==3) { // box kernel
                acc_box[g] += dist*dist;
                if (acc_box[g]>1) {
                    skip[g]=1;
                    break;
                }
            } else if (kernels[q]==4) { // kronecker delta kernel
                if ( (int32_t) dist > 0 ) {
                    skip[g]=1;
                    break;
                }
            }
        }
        
    }
    
    for (m=0; m<ny; m++) { // loop through all test points
        
        // initialize accumulators
        acc_gauss_m = 0;
        acc_epa_m = 0;
        acc_vonmises_m = 0;
        acc_box_m = 0;
        
        for (d2=0; d2<ndimy; d2++) { // loop through test dimensions
            
            d = d2+ndimx;
            
            ix = d2*ny + m; //TODO: optimize (get rid of multiplication)
            idata = d*npoints + n; //TODO: optimize (get rid of multiplication)
            
            if (nlut[d]>0) {
                // look up distance in table
                dist = distance[d][ (int32_t) y[ix] + nlut[d] * (int32_t) data[idata] ];
            } else {
                // compute distance
                dist = y[ix] - data[idata];
            }
            
            if (kernels[d]!=2) {
                dist /= bandwidths[d]; //TODO: optimize (by normalizing data to bandwidth ahead of time)
            }
            
            if (kernels[d]==0) { // gaussian kernel
                acc_gauss_m += dist*dist;
                if (acc_gauss_m>cutoff_gauss) {
                    goto nextm;
                }
            } else if (kernels[d]==1) { // epanechnikov kernel
                acc_epa_m += dist*dist;
                if (acc_epa_m>1) {
                    goto nextm;
                }
            } else if (kernels[d]==2) { // von mises kernel
                acc_vonmises_m += bandwidths[d] * cos(dist);
                //TODO: truncated von mises kernel
            } else if (kernels[d]==3) { // box kernel
                acc_box_m += dist*dist;
                if (acc_box_m>1) {
                    goto nextm;
                }
            } else if (kernels[d]==4) { // kronecker delta kernel
                if ( (int32_t) dist > 0 ) {
                    goto nextm;
                }
            }
            
        }
        
        for (g=0; g<nx; g++) { // loop through all grid points
        
            if (skip[g]==1 || acc_gauss[g]+acc_gauss_m>cutoff_gauss || acc_epa[g]+acc_epa_m>1 || acc_box[g]+acc_box_m>1) {
                continue;
            }
            // accumulate contributions of data points
            output[g+m*nx] += weights[n] * exp( -0.5*(acc_gauss[g]+acc_gauss_m) + acc_vonmises[g] + acc_vonmises_m ) * (1-acc_epa[g]-acc_epa_m);
            
        }
        
        nextm:
            ;
    }

}

free(skip);
free(acc_gauss);
free(acc_epa);
free(acc_vonmises);
free(acc_box);

}
