#include "covariance.h"
#include <string.h>
#include <math.h>
#include <assert.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_sf_erf.h>

#define LOG2PI 1.83787706640935
#define TWOPI 6.283185307179586

CovarianceMatrix* covariance_create( uint16_t ndim ) {
    
    assert (ndim>0);
    
    CovarianceMatrix* m = (CovarianceMatrix*) malloc( sizeof(CovarianceMatrix) );
    m->refcount = 1;
    
    m->ndim = ndim;
    
    m->data = (double*) malloc( ndim*ndim*sizeof(double) );
    
    m->lu.matrix = m->icov.matrix = m->ichol.matrix = NULL;
    m->lu.permutation = NULL;
    m->sigma.points = NULL;
    
    covariance_invalidate_cache( m );
    
    return m;
}

CovarianceMatrix* covariance_create_empty( uint16_t ndim ) {
    return covariance_create( ndim );
}

CovarianceMatrix* covariance_create_zero( uint16_t ndim ) {
    CovarianceMatrix *m = covariance_create( ndim );
    covariance_set_zero( m );
    return m;
}

CovarianceMatrix* covariance_marginalize( CovarianceMatrix* m, uint16_t ndim, uint16_t* dim) {
    
    uint16_t j,k;
    
    assert( ndim<=m->ndim );
    
    CovarianceMatrix* c = covariance_create_empty( ndim );
    
    for (j=0; j<ndim; j++) {
        for (k=0; k<ndim; k++ ) {
            c->data[j*ndim+k] = m->data[ dim[j] * m->ndim + dim[k] ];
        }
    }
    
    return c;
}

void covariance_delete( CovarianceMatrix* m) {
    
    if (m)
    {
        m->refcount--;
        
        if (m->refcount<1)
        {
            covariance_free_cache(m);
            if (m->data) { free(m->data); }
            free(m);
            m = NULL;
        }
    }
    
}

CovarianceMatrix* covariance_use( CovarianceMatrix* m ) {
    
    if (m) { m->refcount++; }
    return m;
}

int covariance_save( CovarianceMatrix* m, FILE* out ) {
    
    if (out==NULL || m==NULL) { return 0; }
    
    if (fwrite( &m->ndim, sizeof(uint16_t), 1, out )==0) { return 0; }
    
    //for future use
    uint8_t save_cache = 0;
    fwrite( &save_cache, sizeof(uint8_t), 1, out );
    
    if (fwrite( m->data, sizeof(double), m->ndim*m->ndim, out )==0) {return 0; }
    
    return 1;
    
}

CovarianceMatrix* covariance_load( FILE* in ) {
    
    CovarianceMatrix* m = NULL;
    
    if (in==NULL) {return NULL;}
    
    uint16_t ndim;
    if ( fread( &ndim, sizeof(uint16_t), 1, in ) != 1 ) {} //TODO
    
    //for future use
    uint8_t load_cache;
    if ( fread( &load_cache, sizeof(uint8_t), 1, in ) != 1 ) {} //TODO
    
    m = covariance_create( ndim );
    
    if ( fread( m->data, ndim*ndim * sizeof(double), 1, in ) != 1) {} //TODO
    
    return m;
    
}

void covariance_invalidate_cache( CovarianceMatrix* m ) {
    
    m->lu.is_outdated = m->icov.is_outdated = m->ichol.is_outdated = m->sigma.is_outdated = m->scaling_factor_is_outdated = 1;
    
}

void covariance_set_zero( CovarianceMatrix *m ) {
    
    memset( (void*) m->data, 0, m->ndim*m->ndim*sizeof(double));
    covariance_invalidate_cache(m);
    
}

void covariance_set_identity( CovarianceMatrix* m ) {
    
    covariance_set_diagonal_uniform( m, 1.0 );
}

void covariance_set_full( CovarianceMatrix* m, double* values ) {
    
    memcpy( (void*) m->data, (void*) values, (size_t) m->ndim*m->ndim*sizeof(double) );
    covariance_invalidate_cache(m);
    
}

void covariance_set_diagonal_uniform( CovarianceMatrix* m, double value) {
    
    uint16_t i;
    
    if (m)
    {
        memset( (void*) m->data, 0, m->ndim*m->ndim*sizeof(double));
        
        for (i=0; i<m->ndim; i++)
        {
            m->data[i*m->ndim + i] = value;
        }
        
        covariance_invalidate_cache(m);
        
    }
    
}

void covariance_set_diagonal( CovarianceMatrix* m, double* value) {
    
    uint16_t i;
    
    if (m)
    {
        memset( (void*) m->data, 0, m->ndim*m->ndim*sizeof(double));
        
        for (i=0; i<m->ndim; i++)
        {
            m->data[i*m->ndim + i] = value[i];
        }
        
        covariance_invalidate_cache(m);
        
    }
    
}

void covariance_free_cache( CovarianceMatrix* m ) {
    
    if (m)
    {
        if (m->lu.matrix != NULL) { gsl_matrix_free( m->lu.matrix ); m->lu.matrix = NULL; }
        if (m->icov.matrix != NULL) { gsl_matrix_free( m->icov.matrix ); m->icov.matrix = NULL; }
        if (m->ichol.matrix != NULL) { gsl_matrix_free( m->ichol.matrix ); m->ichol.matrix = NULL; }
        if (m->lu.permutation != NULL) { gsl_permutation_free( m->lu.permutation ); m->lu.permutation = NULL; }
        if (m->sigma.points != NULL) { free( m->sigma.points ); m->sigma.points = NULL; }
        
        m->lu.signum = 0;
        m->ichol.logconstant = 0;
        
        m->diag_scaling_factor = 0;
        
        covariance_invalidate_cache(m);

    }
}

void covariance_update_diag_scaling_factor(CovarianceMatrix* m) {
    
    uint16_t k;
    double tmp;
    
    tmp = 1;
    
    for (k=0; k<m->ndim; k++) {
        tmp *= m->data[k*m->ndim+k];
    }
    
    m->diag_scaling_factor = 1.0 / sqrt( pow( TWOPI, m->ndim ) * tmp );
    m->diag_scaling_factor /= pow( -gsl_sf_erf( -CUTOFF / sqrt(2) ), m->ndim );
    
    m->scaling_factor_is_outdated = 0;
}

void covariance_alloc_lu(CovarianceMatrix* m) {
    
    if (m->lu.matrix == NULL)
    {
        m->lu.matrix = gsl_matrix_alloc( m->ndim, m->ndim );
        m->lu.permutation = gsl_permutation_alloc( m->ndim );
    }
}

void covariance_update_lu( CovarianceMatrix* m ) {
    
    if (m && m->lu.is_outdated)
    {
        //allocate memory if needed
        covariance_alloc_lu(m);
        
        //copy covariance matrix
        gsl_matrix_view c = gsl_matrix_view_array( m->data, m->ndim, m->ndim );
        gsl_matrix_memcpy( m->lu.matrix, &c.matrix );
        
        //compute LU decomposition
        gsl_linalg_LU_decomp( m->lu.matrix, m->lu.permutation, &m->lu.signum );
        
        m->lu.is_outdated = 0;
    }
}

void covariance_alloc_icov( CovarianceMatrix* m ) {
    
    if (m->icov.matrix == NULL)
    {
        m->icov.matrix = gsl_matrix_alloc( m->ndim, m->ndim );
    }
}

void covariance_update_icov( CovarianceMatrix* m ) {
    
    if (m && m->icov.is_outdated) {
        //allocate memory if needed
        covariance_alloc_icov( m );
        
        //update LU decomposition
        covariance_update_lu( m );
        
        //compute inverse
        gsl_linalg_LU_invert( m->lu.matrix, m->lu.permutation, m->icov.matrix );
        
        m->icov.is_outdated = 0;
    }
}

void covariance_alloc_ichol( CovarianceMatrix* m ) {
    
    if (m->ichol.matrix == NULL)
    {
        m->ichol.matrix = gsl_matrix_alloc( m->ndim, m->ndim );
    }
}

void covariance_update_ichol( CovarianceMatrix* m ) {
    
    uint16_t i;
    
    if (m && m->ichol.is_outdated) {
        //allocate memory if needed
        covariance_alloc_ichol( m );
        
        //update inverse decomposition
        covariance_update_icov( m );
        
        //copy inverse matrix
        gsl_matrix_memcpy( m->ichol.matrix, m->icov.matrix );
        
        //compute cholesky decomposition
        gsl_linalg_cholesky_decomp( m->ichol.matrix );
        
        //compute log constant
        m->ichol.logconstant=0;
        for (i=0; i<m->ndim; i++)
        {
            m->ichol.logconstant += log( gsl_matrix_get( m->ichol.matrix, i, i ) );
        }
        
        m->ichol.logconstant -= 0.5*m->ndim*LOG2PI;
        
        m->ichol.is_outdated = 0;
    }
}

void covariance_alloc_sigma( CovarianceMatrix* m ) {
    
    if (m->sigma.points == NULL)
    {
        m->sigma.k = 3-m->ndim;
        if (m->sigma.k<0) {m->sigma.k = 0;}
        m->sigma.n = 2*m->ndim + (int) (m->sigma.k>0);
        m->sigma.points = (double*) malloc( m->sigma.n * m->ndim * sizeof(double) );
    }
}

void covariance_update_sigma( CovarianceMatrix *m ) {
    
    uint32_t i,j;
    
    if (m && m->sigma.is_outdated) {
        
        covariance_alloc_sigma( m );
        
        gsl_matrix_view view = gsl_matrix_view_array( m->data, m->ndim, m->ndim);
        gsl_matrix *U = gsl_matrix_alloc( m->ndim, m->ndim);
        gsl_matrix *V = gsl_matrix_alloc( m->ndim, m->ndim );
        gsl_vector *S = gsl_vector_alloc( m->ndim );
        gsl_vector *work = gsl_vector_alloc( m->ndim );
        
        gsl_matrix_memcpy( U, &view.matrix );
        
        gsl_linalg_SV_decomp( U, V, S, work );
        
        for (i=0; i<m->ndim; i++)
            gsl_vector_set( S, i, sqrt(gsl_vector_get(S,i))*sqrt(m->ndim+m->sigma.k) );
        
        for (i=0; i<m->ndim; i++)
        {
            for (j=0; j<m->ndim; j++)
            {
                gsl_matrix_set( U, i, j, gsl_matrix_get(U, i, j)*gsl_vector_get(S,j));
                m->sigma.points[j*m->ndim + i] = gsl_matrix_get(U,i,j); // +mean[i] 
                m->sigma.points[(j+m->ndim)*m->ndim + i] = -gsl_matrix_get(U,i,j); // +mean[i] 
            }
        }
        
        if (m->sigma.k>0)
        {
            for (j=0; j<m->ndim; j++)
            {
                m->sigma.points[(2*m->ndim*m->ndim)+j] = 0; //+mean[j];
            }
        }
        
        gsl_matrix_free(U);
        gsl_matrix_free(V);
        gsl_vector_free(S);
        gsl_vector_free(work);
        
        m->sigma.is_outdated = 0;
        
    }
}

void covariance_update_cache( CovarianceMatrix* m ) {
    
    //update cholesky decomposition, will in turn update inverse matrix and LU decomposition
    covariance_update_ichol( m );
    covariance_update_sigma( m );
    covariance_update_diag_scaling_factor( m );
}

LU* covariance_get_lu( CovarianceMatrix* m) {
    covariance_update_lu( m );
    return &m->lu;
}

ICov* covariance_get_icov( CovarianceMatrix* m) {
    covariance_update_icov( m );
    return &m->icov;
}

IChol* covariance_get_ichol( CovarianceMatrix* m) {
    covariance_update_ichol( m );
    return &m->ichol;
}

SigmaPoints* covariance_get_sigmapoints( CovarianceMatrix* m) {
    covariance_update_sigma( m );
    return &m->sigma;
}

void covariance_print( CovarianceMatrix* m ) {
    
    uint16_t i,j;
    
    if (m)
    {
        printf("covariance = [ ");
        for (i=0; i<m->ndim; i++)
        {
            for(j=0; j<m->ndim; j++)
            {
                printf("%3.2g ", m->data[i*m->ndim+j]);
            }
        
            if (i==m->ndim-1)
            {
                printf("]\n");
            } else {
                printf("\n               ");
            }
        }
    }
}



//uint8_t covariance_copy( CovarianceMatrix *destination, CovarianceMatrix *source, uint8_t copycache)
//{
    //if (destination && source && destination->ndim==source->ndim)
    //{
        //memcpy(destination->data, source->data, destination->ndim*destination->ndim*sizeof(double));
        
        //if (copycache)
        //{
            //if (!source->cache.LU_is_outdated)
            //{
                //covarariance_alloc_LU(destination);
                //gsl_matrix_memcpy( destination->cache.LU, source->cache.LU );
                //gsl_permutation_memcpy( destination->cache.LU_perm, source->cache.LU_perm );
                //destination->cache.LU_signum = source->cache.LU_signum;
                //destination->cache.LU_is_outdated = 0;
            //} else { destination->cache.LU_is_outdated = 1; }
            
            //if (!source->icov_is_outdated)
            //{
                //covariance_alloc_icov(destination);
                //gsl_matrix_memcpy( destination->cache.icov, source->cache.icov );
                //destination->cache.icov_is_outdated = 0;
            //} else { destination->cache.icov_is_outdated = 1; }
            
            //if (!source->iS_is_outdated)
            //{
                //covariance_alloc_cholesky(destination);
                //gsl_matrix_memcpy( destination->cache.iS, source->cache.iS );
                //destination->cache.logconstant = source->cache.logconstant;
                //destination->cache.iS_is_outdated = 0;
            //} else { destination->cache.iS_is_outdated = 1; }
            
        //} else {
            //destination->cache.LU_is_outdated = destination->cache.icov_is_outdated = destination->cache.iS_is_outdated = 1;
        //}
        
        //return 1;
    //}
    
    //return 0;
//}
