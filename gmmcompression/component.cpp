#include <string.h>
#include <math.h>
#include "component.h"
#include "covariance.h"
#include <assert.h>

#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>

GaussianComponent* component_create( uint16_t ndim, double *mean, CovarianceMatrix* cov, double weight ) {
    
    GaussianComponent* c = component_create_empty( ndim );
    
    c->weight = weight;
    
    if (mean==NULL) {
        memset( (void*) c->mean, 0, (size_t) ndim*sizeof(double) );
    } else{
        memcpy( (void*) c->mean, (void*) mean, (size_t) ndim*sizeof(double) );
    }
    
    if (cov==NULL) {
        c->covariance = covariance_create_zero( ndim );
        covariance_set_identity( c->covariance );
    } else {
        assert( ndim==cov->ndim );
        c->covariance = covariance_use(cov);
    }
    
    return c;
}

GaussianComponent* component_create_zero( uint16_t ndim ) {
    
    GaussianComponent *c = component_create_empty( ndim );
    
    memset( (void*) c->mean, 0, (size_t) ndim*sizeof(double) );

    c->covariance = covariance_create_zero( ndim );
    
    return c;
}

GaussianComponent* component_create_empty( uint16_t ndim ) {
    
    assert (ndim>0);
    
    GaussianComponent* c = (GaussianComponent*) malloc( sizeof(GaussianComponent) );
    c->refcount = 1;
    
    c->ndim = ndim;
    c->weight = 0;
    
    c->mean = (double*) malloc( ndim*sizeof(double) );

    c->covariance = NULL;
    //covariance_create( ndim );
    
    return c;
}

GaussianComponent* component_marginalize( GaussianComponent* c, uint16_t ndim, uint16_t* dim) {
    
    uint16_t k;
    
    assert( ndim<=c->ndim);
    
    GaussianComponent* z = component_create_empty( ndim );
    
    z->covariance = covariance_marginalize( c->covariance, ndim, dim );
    
    for (k=0; k<ndim; k++) {
        z->mean[k] = c->mean[ dim[k] ];
    }
    
    z->weight = c->weight;
    
    return z;
    
}

void component_delete( GaussianComponent* c) {
    
    if (c)
    {
        c->refcount--;
        
        if (c->refcount<1)
        {
            if (c->mean) { free(c->mean); c->mean=NULL; }
            covariance_delete( c->covariance );
            free(c);
            c = NULL;
        }
    }
}

GaussianComponent* component_use( GaussianComponent* c ) {
    if (c) { c->refcount++; }
    return c;
}

int component_save( GaussianComponent* c, FILE* out ) {
    
    if (out==NULL || c==NULL) { return 0; }
    
    if (fwrite( &c->ndim, sizeof(uint16_t), 1, out )==0) { return 0; }
    if (fwrite( &c->weight, sizeof(double), 1, out )==0) { return 0; }
    if (fwrite( c->mean, sizeof(double), c->ndim, out )==0) { return 0; }
    
    if (covariance_save( c->covariance, out )==0) {return 0;}
    
    return 1;
    
}

GaussianComponent* component_load( FILE* in ) {
    
    GaussianComponent* c = NULL;
    
    if (in==NULL) {return NULL;}
    
    uint16_t ndim;
    if ( fread( &ndim, sizeof(uint16_t), 1, in ) != 1 ) {} //TODO
    
    double weight;
    if( fread( &weight, sizeof(double),1, in ) != 1 ) {} //TODO
    
    double* mean = (double*) malloc( ndim*sizeof(double) );
    if ( fread( mean, sizeof(double), ndim, in ) != ndim ) {} //TODO
    
    CovarianceMatrix* cov = covariance_load( in );
    
    c = component_create( ndim, mean, cov, weight );
    
    covariance_delete( cov );
    free(mean);
    
    return c;
}

void component_set_zero( GaussianComponent *c ) {
    
    component_set_weight( c, 0.0 );
    component_set_mean_uniform( c, 0.0 );
    component_set_covariance_zero( c );
        
}

void component_set_weight( GaussianComponent *c, double val) {
    c->weight = val;
}

void component_set_mean( GaussianComponent *c, double *val) {
    memcpy( (void*) c->mean, (void*) val, (size_t) c->ndim*sizeof(double) );
}

void component_set_mean_uniform( GaussianComponent *c, double val) {
    uint16_t i;
    for (i=0; i<c->ndim; i++) { c->mean[i] = val; }
}

void component_get_mean( GaussianComponent *c, double* result ) {
    
    uint16_t i;
    
    for (i=0; i<c->ndim; i++) {
        result[i] = c->mean[i];
    }
    
}

void component_get_covariance_array( GaussianComponent *c, double* result ) {
    
    memcpy( (void*) result, (void*) c->covariance->data, c->ndim*c->ndim*sizeof(double) );
    
}

double component_get_weight( GaussianComponent *c ) { return c->weight; }

void component_set_covariance( GaussianComponent *c, CovarianceMatrix *cov) {
    
    assert( cov->ndim == c->ndim );
    
    if (c->covariance) { covariance_delete( c->covariance ); c->covariance=NULL; }
    
    c->covariance = covariance_use( cov );
}

CovarianceMatrix* component_get_covariance( GaussianComponent* c ) { return c->covariance; }

void component_set_covariance_zero ( GaussianComponent *c ) {
    
    if (c->covariance==NULL) { c->covariance = covariance_create_empty( c->ndim ); }
    
    covariance_set_zero( c->covariance );
}

void component_set_covariance_identity( GaussianComponent *c ) {
    
    if (c->covariance==NULL) { c->covariance = covariance_create_empty( c->ndim ); }
    
    covariance_set_identity( c->covariance );
}

void component_set_covariance_diagonal( GaussianComponent *c, double *val) {
    
    if (c->covariance==NULL) { c->covariance = covariance_create_empty( c->ndim ); }
    
    covariance_set_diagonal( c->covariance, val );
}

void component_set_covariance_diagonal_uniform( GaussianComponent *c, double val ) {
    
    if (c->covariance==NULL) { c->covariance = covariance_create_empty( c->ndim ); }
    
    covariance_set_diagonal_uniform( c->covariance, val );
}

void component_set_covariance_full( GaussianComponent *c, double* val ) {
    
    if (c->covariance==NULL) { c->covariance = covariance_create_empty( c->ndim ); }
    
    covariance_set_full( c->covariance, val );
}


void component_split( GaussianComponent *component, GaussianComponent **out1, GaussianComponent **out2) {
    
    //create new components
    *out1 = component_create_zero( component->ndim );
    *out2 = component_create_zero( component->ndim );
    
    component_split_inplace( component, *out1, *out2 );
}

void component_split_inplace(GaussianComponent *component, GaussianComponent *out1, GaussianComponent *out2) {
    
    gsl_matrix * U = gsl_matrix_alloc(component->ndim, component->ndim);
    gsl_matrix * V = gsl_matrix_alloc(component->ndim, component->ndim);
    gsl_vector * S = gsl_vector_alloc(component->ndim);
    gsl_vector * work = gsl_vector_alloc(component->ndim);
    
    //copy covariance matrix into U
    gsl_matrix_view cov0 = gsl_matrix_view_array(component->covariance->data, component->ndim, component->ndim);
    gsl_matrix_memcpy(U, (const gsl_matrix *) &cov0.matrix);
    
    //compute singular value decomposition
    gsl_linalg_SV_decomp( U, V, S, work );
    
    //find index of maximum in vector S
    size_t idx = gsl_vector_max_index( S );
    
    //scale corresponding column in V by sqrt of maximum value in S
    //we won't use V anymore, so OK to do scaling in-place
    gsl_vector_view delta = gsl_matrix_column(V, idx);
    gsl_vector_scale( &delta.vector, sqrt( gsl_vector_get( S, idx ) )*0.5 );
    
    //create views on mean
    gsl_vector_view mean1 = gsl_vector_view_array( out1->mean, out1->ndim );
    gsl_vector_view mean2 = gsl_vector_view_array( out2->mean, out2->ndim );
    gsl_vector_view mean0 = gsl_vector_view_array( component->mean, component->ndim );
    
    //compute new mean
    gsl_vector_memcpy( &mean1.vector, &mean0.vector );
    gsl_vector_memcpy( &mean2.vector, &mean0.vector );
    gsl_vector_add( &mean1.vector, &delta.vector );
    gsl_vector_sub( &mean2.vector, &delta.vector );
    
    //create views on covariance, initialize to original covariance
    gsl_matrix_view cov1 = gsl_matrix_view_array( out1->covariance->data, out1->ndim, out1->ndim );
    gsl_matrix_view cov2 = gsl_matrix_view_array( out2->covariance->data, out2->ndim, out2->ndim );
    gsl_matrix_memcpy( &cov1.matrix, &cov0.matrix );
    
    //compute new covariances
    gsl_blas_dger( 1.0, &mean0.vector, &mean0.vector, &cov1.matrix );
    gsl_blas_dger( -0.5, &mean1.vector, &mean1.vector, &cov1.matrix );
    gsl_blas_dger( -0.5, &mean2.vector, &mean2.vector, &cov1.matrix );
    
    gsl_matrix_memcpy( &cov2.matrix, &cov1.matrix );
    
    out1->weight = out2->weight = 0.5*component->weight;
    
    //TODO:
    //component_mean_changed(out1);
    //component_mean_changed(out2);
    //component_covariance_changed(out1);
    //component_covariance_changed(out2);
    
    gsl_matrix_free(U);
    gsl_matrix_free(V);
    gsl_vector_free(S);
    gsl_vector_free(work);
}

//component_update_sigmapoints( GaussianComponent* );
//component_update_cache( GaussianComponent* );

void component_print( GaussianComponent* component ) {
    
    uint16_t i;
    
    if (component)
    {
        printf("refcount = %" PRId32 "\n",component->refcount);
        
        printf("weight = %g\n", component->weight);
        
        if (component->mean) {
            printf("mean = [ ");
            for (i=0; i<component->ndim; i++)
                printf("%g ", component->mean[i]);
            printf("]\n");
        } else {
            printf("error: mean uninitialized\n");
        }
        
        if (component->covariance) {
            covariance_print( component->covariance );
        } else {
            printf("error: covariance uninitialized\n");
        }
    } else {
        
        printf("component is NULL\n" );
    }
    
    printf("\n");
}
