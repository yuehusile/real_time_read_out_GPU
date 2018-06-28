#include "mixture.h"
#include <assert.h>
#include <string.h>
#include <iostream>
#include <math.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>

#include "covariance.h"


#include <stdint.h>

#define cast_uint32_t (uint32_t)

static inline float
fastpow2 (float p)
{
  float offset = (p < 0) ? 1.0f : 0.0f;
  float clipp = (p < -126) ? -126.0f : p;
  int w = clipp;
  float z = clipp - w + offset;
  union { uint32_t i; float f; } v = { cast_uint32_t ( (1 << 23) * (clipp + 121.2740575f + 27.7280233f / (4.84252568f - z) - 1.49012907f * z) ) };

  return v.f;
}

static inline float
fastexp (float p)
{
  return fastpow2 (1.442695040f * p);
}

void print_gsl_matrix( gsl_matrix* m ) {
    
    int nrows = m->size1;
    int ncols = m->size2;
    
    int r,c;
    
    if (m) {
        printf("matrix = [ ");
        for (r=0; r<nrows; r++) {
            for(c=0; c<ncols; c++) {
                printf("%3.2g ", gsl_matrix_get( m, r,c ) );
            }
        
            if (r==nrows-1) {
                printf("]\n");
            } else {
                printf("\n           ");
            }
        }
    }
}


Mixture* mixture_create( uint16_t ndim ) {
    
    assert (ndim>0);
    
    //allocate mixture
    Mixture* m = (Mixture*) malloc( sizeof(Mixture) );
    m->refcount = 1;
    
    //initialize
    m->ndim = ndim;
    m->ncomponents = 0;
    m->sum_n = 0;
    m->sum_weights = 0;
    
    m->buffersize = 0;
    
    m->components = NULL;
    
    //create sample covariance matrix
    m->samplecovariance = covariance_create_empty( ndim );
    covariance_set_identity( m->samplecovariance );
    
    return m;

}

void mixture_delete( Mixture* mixture ) {
    
    if (mixture)
    {
        mixture->refcount--;
        
        if (mixture->refcount<1) {
            
            //delete sample covariance matrix
            covariance_delete( mixture->samplecovariance );
            mixture->samplecovariance = NULL;
            
            //delete components
            mixture_remove_all_components( mixture );
            
            //free buffer
            if (mixture->components) { free( mixture->components ); mixture->components = NULL; }
            
            //free mixture
            free(mixture);
            mixture = NULL;
        }
        
    }

}

void mixture_grow_buffer( Mixture* mixture, uint32_t n) {
    
    if (mixture && n>0)
    {
        mixture->components = (GaussianComponent**) realloc( (void*) mixture->components, (mixture->buffersize+n)*sizeof(GaussianComponent*) );
        mixture->buffersize += n;
    }
}

void mixture_shrink_buffer( Mixture* mixture ) {
    
    if (mixture && mixture->components && mixture->buffersize>mixture->ncomponents)
    {
        if (mixture->ncomponents==0)
        {
            free( mixture->components );
            mixture->components = NULL;
            mixture->buffersize = 0;
        } else {
            mixture->components = (GaussianComponent**) realloc( (void*) mixture->components, mixture->ncomponents * sizeof(GaussianComponent*) );
            mixture->buffersize = mixture->ncomponents;
        }
    }
    }

Mixture* mixture_use( Mixture* mixture ) {
    if (mixture) { mixture->refcount++; }
    return mixture;
}


void mixture_remove_all_components( Mixture* mixture ) {
    uint32_t i;
    
    if (mixture && mixture->components) {
        for (i=0; i<mixture->ncomponents; i++) {
            component_delete( mixture->components[i] );
            mixture->components[i] = NULL;
        }
        mixture->ncomponents = 0;
    }

}

uint32_t mixture_add_component( Mixture* mixture, GaussianComponent* comp ) {
    
    uint32_t retval = mixture->ncomponents;
    
    assert( mixture->ndim == comp->ndim );
    
    if (mixture->buffersize < (mixture->ncomponents+1))
    {
        mixture_grow_buffer( mixture, DEFAULTBUFFERSIZE );
    }
    
    mixture->components[mixture->ncomponents] = component_use( comp );
    mixture->ncomponents++;
    
    return retval;
    
}

uint32_t mixture_add_multiple_components( Mixture* mixture, GaussianComponent** comp, uint32_t ncomp ) {
    
    uint32_t i;
    uint32_t retval = mixture->ncomponents;
    
    if (mixture->buffersize < (mixture->ncomponents+ncomp))
    {
        mixture_grow_buffer( mixture, ncomp );
    }
    
    for (i=0; i<ncomp; i++ ) {
        
        assert( comp[i]->ndim == mixture->ndim );
        
        mixture->components[mixture->ncomponents] = component_use( comp[i] );
        mixture->ncomponents++;
    }
    
    return retval;
    
}

Mixture* mixture_marginalize( Mixture* mixture, uint16_t ndim, uint16_t* dim) {
    
    uint32_t k;
    Mixture* m;
    GaussianComponent* c;
    
    assert( ndim<=mixture->ndim);
    
    m = mixture_create( ndim );
    
    for (k=0; k<mixture->ncomponents; k++) {
        
        c = component_marginalize( mixture->components[k], ndim, dim );
        //c = component_create_zero( ndim );
        mixture_add_component( m, c );
        component_delete( c );
        
    }
    
    return m;
    
}

void mixture_remove_component( Mixture* mixture, GaussianComponent* comp ) {
    
    int32_t index = mixture_find_component( mixture, comp );
    
    if (index>=0) { mixture_remove_components_by_index( mixture, index, 1 ); }
    
}

void mixture_remove_components_by_index( Mixture* mixture, uint32_t index, uint32_t n ) {
    
    uint32_t i;
    
    if ((index+n)<=mixture->ncomponents)
    {
        for (i=0; i<n; i++)
        {
            component_delete( mixture->components[index+i] );
            mixture->components[index+i] = NULL;
        }
        
        memcpy( (void*) &mixture->components[index], (void*) &mixture->components[index+n], (size_t) (mixture->ncomponents - index - n)*sizeof(GaussianComponent**) );
        
        for (i=0; i<n; i++)
        {
            mixture->components[mixture->ncomponents-n+i] = NULL;
        }
        
        mixture->ncomponents-=n;
    }
}

int32_t mixture_find_component( Mixture* mixture, GaussianComponent* comp ) {
    
    uint32_t i;
    int32_t retval = -1;
    
    for (i=0; i<mixture->ncomponents; i++ )
    {
        if ( mixture->components[i] == comp ) {retval = i; break;}
    }
    
    return retval;
}

GaussianComponent* mixture_get_component( Mixture* mixture, uint32_t index ) {
    
    assert( index < mixture->ncomponents );
    
    return mixture->components[index];
}


void mixture_update_cache( Mixture* mixture ) {
    
    uint32_t k;
    
    for (k=0; k<mixture->ncomponents; k++) {
        covariance_update_cache( mixture->components[k]->covariance );
    }
    
}

int mixture_save( Mixture* mixture, FILE* out ) {
    
    uint32_t k;
    
    //check file
    if (out==NULL || mixture==NULL) { return 0; }
    
    //save mixture properties
    if (fwrite( &mixture->ndim, sizeof(mixture->ndim), 1, out )==0) { return 0; }
    if (fwrite( &mixture->ncomponents, sizeof(mixture->ncomponents), 1, out )==0) { return 0; }
    if (fwrite( &mixture->sum_n, sizeof(mixture->sum_n), 1, out )==0) { return 0; }
    if (fwrite( &mixture->sum_weights, sizeof(mixture->sum_weights), 1, out )==0) { return 0; }

    //save sample covariance
    uint8_t b = 0;
    if (mixture->samplecovariance==NULL) {
        fwrite( &b, sizeof(b), 1, out );
    } else {
        b = 1;
        fwrite( &b, sizeof(b), 1, out );
        if (covariance_save( mixture->samplecovariance, out )==0) { return 0; }
    }
    
    //save components
    for (k=0; k<mixture->ncomponents; k++) {
        if (component_save( mixture->components[k], out )==0) {return 0;}
    }
    
    return 1;
    
}

Mixture* mixture_load( FILE* in ) {
    
    Mixture* mixture = NULL;
    uint16_t ndim;
    uint32_t k;
    
    if (in==NULL) {return NULL;}
    
    if (fread( &ndim, sizeof(ndim), 1, in ) != 1 ) {} //TODO
    
    mixture = mixture_create( ndim );
    
    uint32_t ncomponents;
    if ( fread( &ncomponents, sizeof(ncomponents), 1, in ) != 1 ) {} //TODO
    
    double val;
    if ( fread( &val, sizeof(val), 1, in ) !=1 ) {} //TODO
    mixture->sum_n = val;
    if ( fread( &val, sizeof(val), 1, in ) !=1 ) {} //TODO
    mixture->sum_weights = val;
    
    uint8_t b;
    if ( fread( &b, sizeof(b), 1, in ) != 1 ) {} //TODO
    if (b==1) {
        CovarianceMatrix* cov = covariance_load( in );
        mixture_set_samplecovariance( mixture, cov );
        covariance_delete( cov );
    }
    
    if (ncomponents>0) {
        mixture_grow_buffer( mixture, ncomponents );
    }
    
    GaussianComponent* comp;
    
    for (k=0; k<ncomponents; k++ ) {
        comp = component_load( in );
        mixture_add_component( mixture, comp );
        component_delete( comp );
    }
    
    return mixture;
    
}

int mixture_save_to_file( Mixture* mixture, const char* filename ) {
    
    FILE* fid = fopen( filename, "w" );
    
    if (fid==NULL) { return 0; }
    
    int retval = mixture_save( mixture, fid );
    
    fclose( fid ); fid = NULL;
    
    return retval;
}

Mixture* mixture_load_from_file(const char* filename ) {
    
    FILE* fid = fopen( filename, "r" );
    
    if (fid==NULL) { return NULL; }
    
    Mixture* m = mixture_load( fid );
    
    mixture_update_cache(m); 
    
    fclose( fid );
    
    return m;
    
}

Mixture* copy_mixture( Mixture* m ) {
    
    uint32_t k;
    Mixture* mixture = NULL;
    
    mixture = mixture_create( m->ndim );
    mixture->sum_n = m->sum_n;
    mixture->sum_weights = m->sum_weights;
    
    mixture_set_samplecovariance( mixture, m->samplecovariance );
    if ( m->ncomponents > 0 ) {
        mixture_grow_buffer( mixture, m->ncomponents );
    }
    
    for ( k=0; k<m->ncomponents; k++) {
        GaussianComponent* comp = component_create(
            m->ndim, m->components[k]->mean, m->components[k]->covariance, m->components[k]->weight );
        mixture_add_component( mixture, comp );
        component_delete( comp );
    }
     
    mixture_update_cache(m);
    
    return mixture;
}

void mixture_get_means( Mixture* mixture, double* result) {
    
    uint32_t i;
    uint16_t j;
    
    for (i=0; i<mixture->ncomponents; i++) {
        for (j=0; j<mixture->ndim; j++) {
            result[i*mixture->ndim + j] = mixture->components[i]->mean[j];
        }
    }
    
}

void mixture_get_scaling_factors( Mixture* mixture, double* result) {
    
    uint32_t i;
    
    for (i=0; i<mixture->ncomponents; i++) {
        result[i] = mixture->components[i]->covariance->diag_scaling_factor;
    }
    
}

void mixture_get_weights( Mixture* mixture, double* result) {
    
    uint32_t i;
    
    for (i=0; i<mixture->ncomponents; i++) {
        result[i] = mixture->components[i]->weight;
    }
    
}

void mixture_get_covariances( Mixture* mixture, double* result) {
    
    uint32_t i;
    uint16_t nelement;
    
    nelement = mixture->ndim * mixture->ndim;
    
    for (i=0; i<mixture->ncomponents; i++) {
        memcpy( (void*) result, (void*) mixture->components[i]->covariance->data, nelement * sizeof(double) );
        result += nelement;
    }
}


void mixture_set_samplecovariance( Mixture* mixture, CovarianceMatrix* cov ) {

    assert( mixture->ndim == cov->ndim );
    
    covariance_delete( mixture->samplecovariance );
    mixture->samplecovariance = covariance_use( cov );
    
}

CovarianceMatrix* mixture_get_samplecovariance( Mixture* mixture ) {
    return mixture->samplecovariance;
}

double mixture_prepare_weights( Mixture* mixture, uint32_t npoints, double attenuation ) {
    
    uint32_t i;
    
    double attenuated_sum_weights = mixture->sum_weights * attenuation;
    double sum_sample_weights = npoints; //assuming weight per sample of 1
    mixture->sum_weights = attenuated_sum_weights + sum_sample_weights;
    
    double attenuated_sum_n = mixture->sum_n * attenuation;
    mixture->sum_n = attenuated_sum_n + npoints;
    
    double mixing_factor_old = attenuated_sum_weights/mixture->sum_weights;
    double mixing_factor_new = sum_sample_weights/mixture->sum_weights;
    
    double weight = mixing_factor_new / sum_sample_weights;
    
    //adjust weights of existing components
    for (i=0; i<mixture->ncomponents; i++) {
        mixture->components[i]->weight *= mixing_factor_old;
    }
    
    return weight;
    
}

void mixture_addsamples( Mixture* mixture, double* means, uint32_t npoints, uint16_t ndim ) {
    
    uint32_t i;
    double *point;
    GaussianComponent* c;
    
    assert (ndim==mixture->ndim);
    
    //resize buffer if needed
    if ((npoints + mixture->ncomponents) >= mixture->buffersize)
    {
        mixture_grow_buffer( mixture, npoints );
    }
    
    //printf("adding %d samples\n", npoints);
    //printf("current sum of weights: %f\n", mixture->sum_weights );
    
    double weight = mixture_prepare_weights( mixture, npoints, 1.0 );
    
    //double attenuation = 1.0;
    
    //double attenuated_sum_weights = mixture->sum_weights * attenuation;
    //double sum_sample_weights = npoints; //assuming weight per sample of 1
    //mixture->sum_weights = attenuated_sum_weights + sum_sample_weights;
    
    ////printf("new sum of weights: %f\n", mixture->sum_weights );
    ////printf("current sum of n: %f\n", mixture->sum_n );
    
    //double attenuated_sum_n = mixture->sum_n * attenuation;
    //mixture->sum_n = attenuated_sum_n + npoints;
    
    ////printf("new sum of n: %f\n", mixture->sum_n );
    
    //double mixing_factor_old = attenuated_sum_weights/mixture->sum_weights;
    //double mixing_factor_new = sum_sample_weights/mixture->sum_weights;
    
    ////printf("mixing factor old components: %f\n", mixing_factor_old );
    ////printf("mixing factor samples: %f\n", mixing_factor_new );
    
    //double weight = mixing_factor_new / sum_sample_weights;
    
    ////printf("sample weight: %f\n", weight );
    
    ////adjust weights of existing components
    //for (i=0; i<mixture->ncomponents; i++) {
        //mixture->components[i]->weight *= mixing_factor_old;
    //}
    
    //create and add components
    point = means;
    for (i=0; i<npoints; i++)
    {
        c = component_create( ndim, point, mixture->samplecovariance, weight );
        
        mixture->components[mixture->ncomponents] = c;
        mixture->ncomponents++;
        
        point += ndim;
    }
   
    
}


void mixture_evaluate( Mixture* mixture, double* points, uint32_t npoints, double* output) {

    evaluate( mixture->components, mixture->ncomponents, points, npoints, output );
    
}

void mixture_evaluate_diagonal( Mixture* mixture, double* points, uint32_t npoints, double* output) {

    evaluate_diagonal( mixture->components, mixture->ncomponents, points, npoints, output );
    
}

void mixture_evaluategrid( Mixture* mixture, double* grid, uint32_t ngrid, uint16_t ngriddim, uint16_t* griddim, double* points, uint32_t npoints, uint16_t npointsdim, uint16_t* pointsdim, double* output ) {
    
    //TODO: CHECK THIS FUNCTION
    
    uint16_t ndim;
    uint32_t idx;
    uint32_t C, G, Q, D, QD, P;
    double *acc;
    double *grid_acc;
    double dx, tmp, sum;
    double threshold=16; //4 standard deviations
    IChol* ichol=NULL;
    
    
    ndim = ngriddim + npointsdim;
    assert( ndim == mixture->ndim );
    
    //ASSUMPTION: each dimension is only listed once in griddim and pointsdim!!
    
    //allocate grid accumulator array
    grid_acc = (double*) malloc(ngrid * ngriddim * sizeof(double));

    //allocate accumulator and zero its elements
    acc = (double*) malloc( ndim * sizeof(double) );
    
    //zero output array
    memset( output, 0, ngrid*npoints*sizeof(double) );
    
    //loop through all components C
    for (C=0; C<mixture->ncomponents; C++)
    {
        
        ichol = covariance_get_ichol( mixture->components[C]->covariance );
        
        idx = 0; //G*ngriddim+Q
        //loop through grid points G
        for (G=0; G<ngrid; G++)
        {
            //loop through grid dimensions Q
            for (Q=0; Q<ngriddim; Q++)
            {
                dx = grid[idx] - mixture->components[C]->mean[ griddim[Q] ];
                grid_acc[idx]=0;
            
                //loop through first Q dimensions
                for (QD=0; QD<=Q; QD++)
                {
                    grid_acc[G*ngriddim+QD] += dx* gsl_matrix_get( ichol->matrix, griddim[Q], griddim[QD] ); //[Q*ndim+QD];
                }
            
                idx++;
            }
            //printf("grid acc = [%g, %g]\n", grid_acc[G*(ngriddim+ndim)],  grid_acc[G*(ngriddim+ndim)+1] );
        }
        
        idx = 0; //P*npointsdim+D
        // loop through points P
        for (P=0; P<npoints; P++)
        {
            //reset acc
            memset( acc, 0, ndim * sizeof(double) );
            
            //  loop through all dimensions D
            for (D=0; D<npointsdim; D++)
            {
                dx = points[idx] - mixture->components[C]->mean[ pointsdim[D] ];
            
                //loop through all grid dimension
                for (QD=0; QD<ngriddim; QD++)
                {
                    acc[QD] += dx*gsl_matrix_get( ichol->matrix, pointsdim[D], griddim[QD] ); 
                }
                //loop through all dimension D
                for (QD=0; QD<=D; QD++)
                {
                    acc[QD+ngriddim] += dx*gsl_matrix_get( ichol->matrix, pointsdim[D], pointsdim[QD] ); 
                }
                
                idx++;
            }
            
            //printf("acc = [%g, %g]\n", acc[0],  acc[1] );
            
            //loop through all grid points G
            for (G=0; G<ngrid; G++)
            {
                sum = 0;
                
                for (QD=0; QD<ndim; QD++)
                {
                    if (QD<ngriddim)
                    {
                        tmp = grid_acc[G*ngriddim+QD] + acc[QD];
                    } else {
                        tmp = acc[QD];
                    }
                    
                    sum += tmp*tmp;
                    
                }
            
                if (sum<threshold)
                {
                    output[P*ngrid+G] += mixture->components[C]->weight*exp(ichol->logconstant-0.5*sum);
                    //printf( "sum = %f\n", sum );
                }
                
            }
        }
    
    //printf( "log constant = %f\n", ichol->logconstant );
    
    }
    
    
    free(acc);
    free(grid_acc);
    
}

void mixture_evaluategrid_diagonal( Mixture* mixture, double* grid, uint32_t ngrid, uint16_t ngriddim, uint16_t* griddim, double* points, uint32_t npoints, uint16_t npointsdim, uint16_t* pointsdim, double* output ) {
    
    //TODO: CHECK THIS FUNCTION
    
    uint16_t ndim;
    uint32_t idx, out_idx, skip;
    uint32_t C, G, Q, D, P;
    double acc;
    double *grid_acc, *grid_skip;
    double dx, sum;
    double *mean, *cov, scale;
    
    ndim = ngriddim + npointsdim;
    assert( ndim == mixture->ndim );
    
    //ASSUMPTION: each dimension is only listed once in griddim and pointsdim!!
    
    //allocate grid accumulator and skip arrays
    grid_acc = (double*) malloc(ngrid* sizeof(double));
    grid_skip = (double*) malloc(ngrid* sizeof(double));
    
    //zero output array
    memset( output, 0, ngrid*npoints*sizeof(double) );
    
    //loop through all components C
    for (C=0; C<mixture->ncomponents; C++)
    {
        
        scale = mixture->components[C]->covariance->diag_scaling_factor * mixture->components[C]->weight;
        mean = mixture->components[C]->mean;
        cov = mixture->components[C]->covariance->data;
        
        idx = 0; //G*ngriddim+Q
        
        //loop through grid points G
        for (G=0; G<ngrid; G++)
        {
            
            grid_acc[G] = 0;
            grid_skip[G] = 0;
            
            //loop through grid dimensions Q
            for (Q=0; Q<ngriddim; Q++)
            {
                dx = grid[idx+Q] - mean[ griddim[Q] ];
                dx = dx*dx/cov[griddim[Q]*(ndim+1)];
                
                grid_acc[G]+=dx;
                
                if (grid_acc[G]>CUTOFF) {
                    grid_skip[G] = 1;
                    break;
                }
            
            }
            
            idx += ngriddim;
        
        }
        
        idx = 0; //P*npointsdim+D
        out_idx = 0;
        
        // loop through points P
        for (P=0; P<npoints; P++)
        {
            //reset acc
            acc = 0;
            skip = 0;
            
            //  loop through all dimensions D
            for (D=0; D<npointsdim; D++)
            {
                dx = points[idx+D] - mean[ pointsdim[D] ];
                dx = dx*dx/cov[ pointsdim[D]*(ndim+1) ];
                acc += dx;
                
                if (acc>CUTOFF) {
                    skip = 1;
                    break;
                }
            }
            
            if (!skip) {
                
                //loop through all grid points G
                for (G=0; G<ngrid; G++)
                {
                    sum = grid_acc[G] + acc;
                    
                    if ( grid_skip[G] || sum>CUTOFF ) {continue;}
                    
                    output[out_idx+G] += scale*exp(-0.5*sum);
                    
                }
                
            }
            
            idx += npointsdim;
            out_idx += ngrid;
        }
    
    }
    
    free(grid_acc);
    free(grid_skip);
    
}

void mixture_evaluategrid_diagonal_multi( Mixture* mixture, double* grid_acc, uint32_t ngrid, double* points, uint32_t npoints, uint16_t npointsdim, uint16_t* pointsdim, double* output )
{
    uint32_t P;
    double* current_point = points;
    double* current_output = output;
    
    for (P=0; P<npoints; P++)
    {
        mixture_evaluategrid_diagonal( mixture, grid_acc, ngrid, current_point, npointsdim, pointsdim, current_output );
	// for debugging
/*	if (P==0)
	{
	    printf("\nP=%d\n",P);
	    for (int i=0;i<npointsdim;i++)
	        printf("current_point[%d]=%f\n",i,current_point[i]);
	    for (int i=0;i<ngrid;i++)
	        printf("current_output[%d]=%.10e\n",i,current_output[i]);
	}
*/        current_point += npointsdim;
        current_output += ngrid;
    }
    
}

void mixture_evaluategrid_diagonal( Mixture* mixture, double* grid_acc, uint32_t ngrid, double* testpoint, uint16_t npointsdim, uint16_t* pointsdim, double* output ) {
    
    //INPUTS
    
    //mixture
    
    //cached grid_acc for each component
    //a grid_acc array has ngrid entries, negative values mean 'skip'
    
    //single test spike
    
    //ALGORITHM
    
    //for each component
    //compute distance between component and test spike
    //bail out early if necessary
    
    //for each grid point
    //skip if grid_acc<0 or grid_acc+acc>cutoff
    //otherwise compute probability
    uint16_t ndim;
    uint32_t C, D, G;
    
    double scale, acc, dx, sum;
    double *mean, *cov;
    
    double *tmp;
    
    ndim = mixture->ndim;
    
    //loop through all mixture components
    for (C=0; C<mixture->ncomponents; C++) {
        
        mean = mixture->components[C]->mean;
        cov = mixture->components[C]->covariance->data;
        scale = mixture->components[C]->covariance->diag_scaling_factor * mixture->components[C]->weight;
        //scale = mixture->components[C]->covariance->diag_scaling_factor*1000 * mixture->components[C]->weight;
        //scale = mixture->components[C]->weight;
        
        //compute partial squared distance to test spike
        acc = 0;
        for (D=0; D<npointsdim; D++)
        {
            dx = testpoint[D] - mean[ pointsdim[D] ];
            dx = dx*dx/cov[ pointsdim[D]*(ndim+1) ];
            acc += dx;
            
            if (acc>CUTOFF) {
                acc = -1;
                break;
            }
        }
        
        //loop through all gridpoints
        if (acc>=0) {
            
            tmp = &grid_acc[C*ngrid];
            
            //loop through all grid points G
            for (G=0; G<ngrid; G++)
            {
                if (*tmp<0 || (sum = *tmp + acc)>CUTOFF ) { tmp++; continue; }
                
                output[G] += scale*fastexp( (float) -0.5*sum);
                //******************************* temporal change for handling numerical issues
		//******************************* should change at encoding part
		//output[G] += fastexp( (float) -0.5*sum);
                tmp++;
            }
            
        }
    }
/*    for (int g=0;g<ngrid;g++)
    {
        output[g] *= mixture->components[0]->covariance->diag_scaling_factor;
    }*/
}

void mixture_prepare_grid_accumulator( Mixture* mixture, double* grid, uint32_t ngrid, uint16_t ngriddim, uint16_t* griddim, double* grid_acc ) {
    
    double* tmp;
    double *mean, *cov;
    uint16_t ndim;
    uint32_t C, G, Q;
    uint32_t idx;
    double dx;
    
    ndim = mixture->ndim;
    
    //grid_acc = malloc( mixture->ncomponents * ngrid * sizeof(double*) );
    
    for (C=0; C<mixture->ncomponents; C++) {
        
        mean = mixture->components[C]->mean;
        cov = mixture->components[C]->covariance->data;
        
        idx = 0; //G*ngriddim+Q
        
        tmp = &grid_acc[C*ngrid];
        
        //loop through grid points G
        for (G=0; G<ngrid; G++) {
            
            *tmp = 0;
            
            //loop through grid dimensions Q
            for (Q=0; Q<ngriddim; Q++) {
                dx = grid[idx+Q] - mean[ griddim[Q] ];
                dx = dx*dx/cov[griddim[Q]*(ndim+1)];
                
                *tmp+=dx;
                
                if (*tmp>CUTOFF) {
                    *tmp = -1;
                    break;
                }
            
            }
            
            idx += ngriddim;
            tmp++;
        }
        
    }

}


Mixture* mixture_compress( Mixture* mixture, double threshold, uint8_t weighted_hellinger ) {
    
    //linked list of models [index, n, model, distance]
    ModelList modellist;
    modellist.head = NULL;
    modellist.nmodels = 0;
    
    GaussianComponent *c, *c1, *c2;
    GaussianComponent **components, **components1, **components2;
    uint32_t ncomponents, n1, n2;
    double distance, distance1, distance2;
    
    //printf("compressing mixture with %d components\n", mixture->ncomponents);
    double sumw=0;
    unsigned int i;
    for (i=0; i<mixture->ncomponents; i++) { sumw+=mixture->components[i]->weight; }
    //printf("sum of weights = %f\n", sumw);
    
    c = component_create_empty( mixture->ndim );
    
    //fit mixture with single gaussian (moment match)
    moment_match( mixture->components, mixture->ncomponents, c, 1 );
    //compute distance between fit and mixture
    distance = hellinger_single( mixture->components, mixture->ncomponents, c, weighted_hellinger );
    //add modelnode to modellist with index=indices, n=mixture->ncomponents, model=fit, 
    modellist_add( &modellist, mixture->components, mixture->ncomponents, c, distance );
    
    //printf("First model (distance=%f):\n", distance);
    //component_print( c );
    
    //component_print( c );
    //printf("H=%f\n",distance);
    
    uint8_t done;
    
    //uint32_t loopnumber=0;
    uint32_t innerloop=0;
    
    while (modellist.head->distance > threshold)
    {
        //loopnumber++;
        //printf("outer loop %d\n",loopnumber);
        
        //printf("initial split of component (distance=%f)\n", modellist.head->distance );
        
        //get submixture with largest distance to its model
        c = modellist_pop( &modellist, &components, &ncomponents );
        
        //start of two gaussian approximation
        
        //to initialize, split model component in two
        //this function will allocate two new components
        component_split( c, &c1, &c2 );
        //component_print( c1 );
        //component_print( c2 );
        
        done = 0;
        innerloop = 0;
        while (!done && innerloop<100)
        {
            innerloop++;
            
            //compute responsibilities for each mixture component
            done = assign_responsibilities( components, ncomponents, c1, c2, &components1, &components2, &n1, &n2); 
            
            //printf("assigned responsibilities: %d vs %d\n", n1, n2 );
            
            //refit
            moment_match( components1, n1, c1, 1 );
            moment_match( components2, n2, c2, 1 );
            
        }    
        
        //if (innerloop>=50) {printf("inner loop overrun\n");}
        
        //end of two gaussian approximation
        
        //compute hellinger distance
        //distance1 = hellinger_single( mixture, s1->model, s1->index, s1->n );
        //distance2 = hellinger_single( mixture, s2->model, s2->index, s2->n );
        distance1 = hellinger_single( components1, n1, c1, weighted_hellinger );
        distance2 = hellinger_single( components2, n2, c2, weighted_hellinger );
        
        //printf("H1=%f, H2=%f\n",distance1, distance2);
        
        //add to model
        //compressedmixture_add( model, s1, distance1 );
        //compressedmixture_add( model, s2, distance2 );
        modellist_add( &modellist, components1, n1, c1, distance1 );
        modellist_add( &modellist, components2, n2, c2, distance2 );
        
        //printf("Fitted models (distances = %f, %f):\n", distance1, distance2 );
        //component_print(c1);
        //component_print(c2);
        
        //printf("\n");
        
        //free old node
        //submixture_delete( tmp );
        component_delete( c );
    }
    
    //component_print( modellist.head->model );
    
    //convert model into mixture
    Mixture* m = model2mixture( &modellist );
    m->sum_n = mixture->sum_n;
    m->sum_weights = mixture->sum_weights;
    
    //printf("model list n = %d\n", modellist.nmodels );
    
    //component_print( modellist.head->model );
    
    //free model nodes
    ModelNode *prev;
    ModelNode *current = modellist.head;
    while (current) {
        //printf("while deleting model list\n");
        //component_print( current->model );
        component_delete( current->model );
        prev = current;
        current = prev->next;
        free( prev );
    }
    
    //return new mixture
    return m;
    
}

void mixture_merge_samples( Mixture* mixture, double* means, uint32_t npoints, double threshold ) {
    
    GaussianComponent** comp = (GaussianComponent**) malloc( 2*sizeof(GaussianComponent*) );
    GaussianComponent* samplecomp;
    GaussianComponent* mergedcomp;
    int64_t foundcomp = -1;

    uint32_t n, c;
    double distance, min_distance;
    
    double* currentpoint;
    
    double weight = mixture_prepare_weights( mixture, npoints, 1.0 );
    
    for (n=0; n<npoints; n++ ) {
        currentpoint = &means[n*mixture->ndim];
        samplecomp = component_create(mixture->ndim, currentpoint, mixture->samplecovariance, weight);
        comp[0] = samplecomp;

        min_distance = threshold;
        foundcomp = -1;
        
        for (c=0; c<mixture->ncomponents; c++ ) {

            distance = compute_mahalanobis_distance( mixture->components[c], samplecomp->mean );
            if (distance < min_distance) {
                min_distance = distance;
                foundcomp = c;
            }
        }

        if (foundcomp>=0 && min_distance<=threshold) {
            comp[1] = mixture->components[foundcomp];
            mergedcomp = component_create_empty( mixture->ndim );
            moment_match(comp, 2, mergedcomp, 1);

            component_delete( mixture->components[foundcomp] );
            mixture->components[foundcomp] = mergedcomp;

        } else {

            mixture_add_component( mixture, samplecomp );

        }
        
        component_delete(samplecomp);
    }
    
    free(comp);
    
}

void mixture_merge_samples_match_bandwidth( Mixture* mixture, double* means, uint32_t npoints, double threshold ) {
    
    GaussianComponent** comp = (GaussianComponent**) malloc( 2*sizeof(GaussianComponent*) );
    GaussianComponent* samplecomp;
    GaussianComponent* mergedcomp;
    int64_t foundcomp = -1;
    //int found;
    
    uint32_t n, c;
    double distance, min_distance;
    
    double* currentpoint;
    
    double weight = mixture_prepare_weights( mixture, npoints, 1.0 );
    
    for (n=0; n<npoints; n++ ) {
        currentpoint = &means[n*mixture->ndim];
        samplecomp = component_create(mixture->ndim, currentpoint, mixture->samplecovariance, weight );
        comp[0] = samplecomp;

        min_distance = threshold;
        foundcomp = -1;
        
        for (c=0; c<mixture->ncomponents; c++ ) {

            distance = compute_mahalanobis_distance( mixture->components[c], samplecomp->mean );
            if (distance < min_distance) {
                min_distance = distance;
                foundcomp = c;
            }
        }

        if (foundcomp>=0 && min_distance<threshold) {
            comp[1] = mixture->components[foundcomp];
            mergedcomp = component_create_empty( mixture->ndim );
            moment_match_bandwidth(comp, 2, mergedcomp, 1);

            component_delete( mixture->components[foundcomp] );
            mixture->components[foundcomp] = mergedcomp;
            
            covariance_update_diag_scaling_factor( mergedcomp->covariance );

        } else {

            mixture_add_component( mixture, samplecomp );
            
            covariance_update_diag_scaling_factor( samplecomp->covariance );

        }
        
        component_delete(samplecomp);
    }
    
    free(comp);
    
}

void mixture_merge_samples_constant_covariance( Mixture* mixture, double* means, uint32_t npoints, double threshold ) {
    
    GaussianComponent* samplecomp;
    int64_t foundcomp = -1;
    //int found;
    
    uint32_t n, c, k;
    double distance, min_distance, w;
    
    double* currentpoint;
    
    double weight = mixture_prepare_weights( mixture, npoints, 1.0 );
    
    for (n=0; n<npoints; n++ ) {
        currentpoint = &means[n*mixture->ndim];
        
        min_distance = threshold;
        foundcomp = -1;
        
        for (c=0; c<mixture->ncomponents; c++ ) {
            distance = compute_mahalanobis_distance( mixture->components[c], currentpoint );
            if (distance < min_distance) {
                min_distance = distance;
                foundcomp = c;
            }
        }
        
        //printf( "min distance = %f, component = %d\n", min_distance, foundcomp );
        
        if (foundcomp>=0 && min_distance<threshold) {
            w = mixture->components[ foundcomp ]->weight;
            for (k=0; k<mixture->ndim; k++) {
                mixture->components[ foundcomp ]->mean[k] = mixture->components[ foundcomp ]->mean[k] * (w/(w+weight)) + currentpoint[k] * (weight/(w+weight));
            }
            mixture->components[ foundcomp ]->weight += weight;
            
            covariance_update_diag_scaling_factor( mixture->components[ foundcomp ]->covariance );
        } else {
            samplecomp = component_create(mixture->ndim, currentpoint, mixture->samplecovariance, weight );
            covariance_update_diag_scaling_factor( samplecomp->covariance );
            mixture_add_component( mixture, samplecomp );
            component_delete(samplecomp);
        }
        
    }
    
}



void moment_match_bandwidth( GaussianComponent** components, uint32_t ncomponents, GaussianComponent* output, uint8_t normalize ) {
    
    uint32_t i,j,idx;
    uint16_t ndim = output->ndim; //CHECK!!
    
    component_set_zero( output );
    
    for (i=0; i<ncomponents; i++)
    {
        
        output->weight += components[i]->weight;
        
        for (j=0; j<ndim; j++ )
        {
            output->mean[j] +=components[i]->mean[j] * components[i]->weight;
        }
        
        for (j=0; j<ndim; j++ )
        {
            idx = j*ndim+j;
            output->covariance->data[idx] += components[i]->weight * ( components[i]->covariance->data[idx] + components[i]->mean[j]*components[i]->mean[j] );
        }
    }
    
    if (normalize)
    {
        for (j=0; j<ndim; j++) {
            output->mean[j] /= output->weight;
                output->covariance->data[j*ndim+j]/=output->weight;
        }
    }
    
    for (j=0; j<ndim; j++)
    {
        output->covariance->data[j*ndim+j] -= output->mean[j]*output->mean[j];
    }
}

void moment_match( GaussianComponent** components, uint32_t ncomponents, GaussianComponent* output, uint8_t normalize ) {
    
    uint32_t i,j,k,idx;
    uint16_t ndim = output->ndim; //CHECK!!
    
    component_set_zero( output );
    
    for (i=0; i<ncomponents; i++)
    {
        
        output->weight += components[i]->weight;
        
        for (j=0; j<ndim; j++ )
        {
            output->mean[j] +=components[i]->mean[j] * components[i]->weight;
        }
        
        for (j=0; j<ndim; j++ )
        {
            for (k=0; k<ndim; k++)
            {
                idx = j*ndim+k;
                output->covariance->data[idx] += components[i]->weight * ( components[i]->covariance->data[idx] + components[i]->mean[j]*components[i]->mean[k] );
            }
        }
    }
    
    if (normalize)
    {
        for (j=0; j<ndim; j++) {
            output->mean[j] /= output->weight;
            for (k=0; k<ndim; k++) {
                output->covariance->data[j*ndim+k]/=output->weight;
            }
        }
    }
    
    for (j=0; j<ndim; j++)
    {
        for (k=0; k<ndim; k++)
        {
            output->covariance->data[j*ndim+k] -= output->mean[j]*output->mean[k];
        }
    }
}

GaussianComponent* modellist_pop( ModelList* modellist, GaussianComponent*** components, uint32_t* ncomponents) {
    
    if (modellist->head==NULL) {return NULL;}
    
    ModelNode* head = modellist->head;
    
    modellist->head = modellist->head->next;
    
    *components = head->components;
    GaussianComponent* retval = head->model;
    *ncomponents = head->ncomponents;
    
    free(head);
    
    modellist->nmodels--;
    
    return retval;
}

void modellist_add( ModelList* modellist, GaussianComponent** components, uint32_t ncomponents, GaussianComponent* model, double distance ) {
    
    ModelNode *prev, *current;
    
    //create node
    ModelNode *newnode = (ModelNode*) malloc( sizeof(ModelNode) );
    newnode->components = components;
    newnode->ncomponents = ncomponents;
    newnode->model = model;
    newnode->distance = distance;
    
    if (modellist->head==NULL) {
        modellist->head = newnode;
        newnode->next = NULL;
    } else if (modellist->head->distance <= distance) {
        newnode->next = modellist->head;
        modellist->head = newnode;
    } else {
        prev = modellist->head;
        current = prev->next;
        while (1) {
            if (current==NULL || current->distance<=distance) {
                newnode->next = current;
                prev->next = newnode;
                break;
            }
            prev = current;
            current = prev->next;
        }
    }
    
    modellist->nmodels++;
    
}

Mixture* model2mixture( ModelList* modellist ) {
    
    if (modellist->nmodels==0) { return NULL; }
    
    uint16_t ndim = modellist->head->model->ndim;
    
    Mixture *m = mixture_create( ndim );
    mixture_grow_buffer( m, modellist->nmodels);
    
    ModelNode *current = modellist->head;
    
    while (current) {
        //mixture_add_component( m, component_use(current->model) );
        mixture_add_component( m, current->model );
        current = current->next;
    }
    
    return m;
}

uint8_t assign_responsibilities( GaussianComponent **components, uint32_t ncomponents, GaussianComponent *c1, GaussianComponent* c2, GaussianComponent*** components1, GaussianComponent ***components2, uint32_t *n1, uint32_t *n2) {
    
    uint32_t i, newi;
    double distance1, distance2;
    double maxdistance = 0;
    GaussianComponent **current = components;
    GaussianComponent *temp;
    uint32_t swapseq = 0;
    uint32_t maxidx=0;
    uint32_t done;
    
    *n1 = *n2 = 0;
    
    //loop through all selected components
    for (i=0; i<ncomponents; i++)
    {
        //compute distance to node1 and node2
        distance1 = compute_distance( components[ i ], c1 );
        distance2 = compute_distance( components[ i ], c2 );
        
        //printf("distance1=%f, distance2=%f\n", distance1, distance2);
        
        //determine responsibility (node with smallest distance)
        if (distance1<=distance2) {
            *n1+=1;
            swapseq = 0;
            if (distance1 > maxdistance)
            {
                maxdistance = distance1;
                maxidx = i;
            }
            current++;
        } else {
            //swap indices at current and n-n1-1
            newi = ncomponents-1-*n1;
            temp = *current;
            *current = components[newi];
            components[newi] = temp;
            
            *n2+=1;;
            swapseq++;
            
            if (distance2 > maxdistance)
            {
                maxdistance = distance2;
                maxidx = newi;
            }
        }
    }
    
    if (*n1 == 0)
    {
        //swap component with largest distance to front
        newi=0;
        temp = components[maxidx];
        components[maxidx] = components[newi];
        components[newi] = temp;
        
        *n1+=1;
        *n2-=1;
        
        current = &components[1];
        
        done = 1;
        
    } else if (*n2 == 0)
    {
        //swap component with largest distance to end
        newi=ncomponents-1;
        temp = components[maxidx];
        components[maxidx] = components[newi];
        components[newi] = temp;
        
        *n1-=1;
        *n2+=1;
        
        current = &components[newi];
        
        done = 1;
    } else
    {
        done = (uint8_t) swapseq==*n2;
    }
    
    *components1 = components;
    *components2 = current;
    
    return done;
    
}

double compute_mahalanobis_distance( GaussianComponent *c1, double* point ) {
    
    ICov* pICov = covariance_get_icov( c1->covariance );
    double* delta = (double*) malloc( c1->ndim * sizeof(double) );
    
    uint32_t j,k;
    double tmp, distance=0;
    
    for (j=0; j<c1->ndim; j++) { delta[j] = point[j] - c1->mean[j]; }
    
    for (k=0; k<c1->ndim; k++) {
        tmp = 0;
        for (j=0; j<c1->ndim; j++) {
            tmp+=delta[j]*gsl_matrix_get( pICov->matrix, j, k );
        }
        distance += tmp * delta[k];
    }
    
    distance = sqrt(distance);
    
    free(delta);
    
    return distance;
}

double compute_distance( GaussianComponent *c1, GaussianComponent *c2 ) {
    
    uint32_t j,k;
    double *delta = (double*) malloc( c1->ndim*sizeof(double) );
    
    double sum, trace, tmp;
    double det;
    double distance;
    LU* pLU;
    
    //compute determinant of component covariance matrix
    pLU = covariance_get_lu( c1->covariance );
    det = gsl_linalg_LU_det( pLU->matrix, pLU->signum );
    
    IChol* pIChol = covariance_get_ichol( c1->covariance );
    ICov* pICov = covariance_get_icov( c1->covariance );
    
    // compute delta * icov * delta.T using cached cholesky decomposition
    // where delta = c2->mean - c1->mean
    for (j=0; j<c1->ndim; j++) { delta[j] = c2->mean[j] - c1->mean[j]; }
    sum = 0;
    for (k=0; k<c1->ndim; k++) {
        tmp = 0;
        for (j=0; j<c1->ndim; j++) {
            tmp += delta[j]*gsl_matrix_get(pIChol->matrix,j,k);
        }
        sum+=tmp*tmp;
    }
    
    //compute trace of cached_icov * result->components[0]->covariance
    trace = 0;
    for (j=0; j<c1->ndim; j++) {
        for (k=0; k<c1->ndim; k++) {
            trace += gsl_matrix_get( pICov->matrix, j, k ) * c2->covariance->data[k*c1->ndim+j];
        }
        
    }
    
    
    //compute distance
    //gmm_component_update_LU( a );
    
    pLU = covariance_get_lu( c2->covariance );
    distance = 0.5*( log( det / gsl_linalg_LU_det( pLU->matrix, pLU->signum ) ) + trace + sum - c1->ndim );
    
    free(delta);
    
    return distance;
}

void evaluate( GaussianComponent **components, uint32_t ncomponents, double* points, uint32_t npoints, double* output ) {
    
    uint32_t c, p;
    uint16_t d,d2;
    double tmp, acc;
    IChol* ichol = NULL;
    uint16_t ndim = components[0]->ndim;
    double threshold = 16; //4 standard deviations
    
    double *dx = (double*) malloc( ndim*sizeof(double) );
    
    memset( (void*) output, 0, npoints*sizeof(double) );
    
    for (c=0; c<ncomponents; c++) {
        ichol = covariance_get_ichol( components[c]->covariance );
        for (p=0; p<npoints; p++) {
            tmp = 0;
            for (d=0; d<ndim; d++) {
                dx[d] = points[p*ndim+d] - components[c]->mean[d];
            }
            for (d=0; d<ndim; d++) {
                acc=0;
                for (d2=d; d2<ndim; d2++) {
                    acc+=dx[d2]*gsl_matrix_get( ichol->matrix, d2, d );
                }
                tmp+=acc*acc;
            }
            if (tmp<threshold) {
                output[p]+=components[c]->weight * exp( ichol->logconstant - 0.5*tmp );
            }
        }
    }
    
    free(dx);

}

void evaluate_diagonal( GaussianComponent **components, uint32_t ncomponents, double* points, uint32_t npoints, double* output ) {
    
    uint32_t c, p;
    uint16_t d, skip;
    double acc;
    uint16_t ndim = components[0]->ndim;
    
    uint32_t cov_index, point_index;
    
    double dx;
    
    double *cov, *mean, scale;
    
    //clear output array
    memset( (void*) output, 0, npoints*sizeof(double) );
    
    for (c=0; c<ncomponents; c++) {
        
        scale = components[c]->covariance->diag_scaling_factor * components[c]->weight;
        mean = components[c]->mean;
        cov = components[c]->covariance->data;
        
        point_index = 0;
        
        for (p=0; p<npoints; p++) {
            
            acc = 0;
            skip = 0;
            
            cov_index = 0;
            
            for (d=0; d<ndim; d++) {
                
                dx = points[point_index+d] - mean[d];
                dx = dx*dx/cov[cov_index];
                acc += dx;
                
                if (acc>CUTOFF) {
                    skip = 1;
                    break;
                }
                
                cov_index += ndim+1;
            }
            
            
            if (!skip) {
                output[p]+=scale * exp( -0.5*acc ); 
            }
            
            point_index += ndim;
        }
    }

}


#define MAXV 3

double hellinger_single( GaussianComponent** components, uint32_t n, GaussianComponent* model, uint8_t weighted) {
    
    uint32_t c,i,j;
    SigmaPoints *sigmapoints, *modelsigmapoints;
    
    uint16_t ndim, s;
    
    double f0 = 0, f1 = 0, f2 = 0;
    double acc, tmp;
    double *pdf1, *pdf2;
    
    double *w;
    double w0;
    double H;
    
    if (weighted>0) {
        f1 = 1;
        f2 = 1;
        for (i=0; i<n; i++) {
            f0 += components[i]->weight;
        }
        f0 = 0.5*f0 + 0.5*model->weight;
    } else {
        for (i=0; i<n; i++) {
            f1 += components[i]->weight;
        }
        f2 = model->weight;
        f0 = 1;
    }
    
    //printf("f0=%f, f1=%f, f2=%f\n", f0, f1, f2);
    
    ndim = model->ndim;
    
    modelsigmapoints = covariance_get_sigmapoints( model->covariance );
    double *tmpmodelsigma = (double*) malloc( ndim*modelsigmapoints->n * sizeof(double) );
    memcpy( (void*) tmpmodelsigma, (void*) modelsigmapoints->points, ndim*modelsigmapoints->n * sizeof(double) );
    for (i=0; i<modelsigmapoints->n; i++) {
        for (j=0;j<ndim;j++) {
            tmpmodelsigma[i*ndim+j]+=model->mean[j];
        }
    }
        
    //printf("model sigma points:");
    //for (i=0;i<modelsigmapoints->n;i++) { printf("%f ",tmpmodelsigma[i]);}
    //printf("\n");
    
    
    acc=0;
    
    pdf1 = (double*) calloc( modelsigmapoints->n, sizeof(double) );
    pdf2 = (double*) calloc( modelsigmapoints->n, sizeof(double) );
    
    w = (double*) malloc( modelsigmapoints->n * sizeof(double) );
    for (i=0; i<modelsigmapoints->n; i++) {
        w[i] = 1.0/(2.0*(ndim+modelsigmapoints->k));
    }
    if (modelsigmapoints->k>0) { w[modelsigmapoints->n-1] = (double) modelsigmapoints->k / (double) (ndim+modelsigmapoints->k); }
    
    //for (i=0;i<modelsigmapoints->n; i++) {
    //    printf("w[%d] = %f ", i,w[i]);
    //}
    //printf("\n");
    
    double *tmpsigma = (double*) malloc( ndim * modelsigmapoints->n * sizeof(double) );
    
    //loop through all components' sigma points
    for (c=0; c<n; c++) {
        
        //get sigma points
        //add mean (sigma points are zero-centered)
        sigmapoints = covariance_get_sigmapoints( components[c]->covariance );
        memcpy( (void*) tmpsigma, (void*) sigmapoints->points, ndim*sigmapoints->n * sizeof(double) );
        for (i=0; i<sigmapoints->n; i++) {
            for (j=0;j<ndim;j++) {
                tmpsigma[i*ndim+j]+=components[c]->mean[j];
            }
        }
        
        //printf("sigma points:");
        //for (i=0;i<modelsigmapoints->n;i++) { printf("%f ",tmpsigma[i]);}
        //printf("\n");
        
        //evaluate mixture at sigma points
        evaluate( components, n, tmpsigma, sigmapoints->n, pdf1 );
        evaluate( &model, 1, tmpsigma, sigmapoints->n, pdf2 );
        
        tmp = 0;
        for (s=0; s<sigmapoints->n; s++) {
            //printf( "pdf1[%d]=%f, pdf2[%d]=%f, pdf0[%d]=%f\n", s, pdf1[s]/f1, s, pdf2[s]/f2, s, (0.5*( pdf1[s]/f1 + pdf2[s]/f2 )/f0) );
            tmp += w[s] * pow( sqrt( pdf1[s]/f1 ) - sqrt( pdf2[s]/f2 ), 2 ) / (0.5*( pdf1[s]/f1 + pdf2[s]/f2 )/f0);
        }
        
        //printf("n = %d, tmp = %f\n", sigmapoints->n, tmp );
        
        w0 = 0.5*components[c]->weight/(f1*f0);
        acc+= 0.5*w0*tmp;
        
    }
    
    evaluate( components, n, tmpmodelsigma, modelsigmapoints->n, pdf1 );
    evaluate( &model, 1, tmpmodelsigma, modelsigmapoints->n, pdf2 );
    
    tmp = 0;
    for (s=0; s<modelsigmapoints->n; s++) {
        tmp += w[s] * pow( sqrt( pdf1[s]/f1 ) - sqrt( pdf2[s]/f2 ), 2 ) / (0.5*(pdf1[s]/f1 + pdf2[s]/f2)/f0);
    }
    
    w0 = 0.5*model->weight/(f2*f0);
    acc+= 0.5*w0*tmp;
    
    H = sqrt( fabs( acc ) );
    
    free(pdf1);
    free(pdf2);
    free(w);
    free(tmpsigma);
    free(tmpmodelsigma);
    
    return H;
    
}
    

