#ifndef COVARIANCE_H
#define COVARIANCE_H

#include <inttypes.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_permutation.h>

#define CUTOFF 16 //threshold for truncated gaussian kernel
//#define CUTOFF 64 //threshold for truncated gaussian kernel

typedef struct LU_cache_t {
    uint8_t is_outdated;
    int signum;
    gsl_matrix *matrix;
    gsl_permutation *permutation;
} LU;

typedef struct icov_cache_t {
    uint8_t is_outdated;
    gsl_matrix *matrix;
} ICov;

typedef struct iS_cache_t {
    uint8_t is_outdated;
    gsl_matrix *matrix;
    double logconstant;
} IChol;

typedef struct sigma_points_cache_t {
    uint8_t is_outdated;
    double *points;
    uint32_t n; //number of sigma points
    int16_t k;
} SigmaPoints;


typedef struct covariance_matrix_t {
    uint32_t refcount;
    uint16_t ndim;
    double *data;
    double diag_scaling_factor; //to make it a proper distribution that integrates to one
    uint8_t scaling_factor_is_outdated;
    LU lu;
    ICov icov;
    IChol ichol;
    SigmaPoints sigma;
} CovarianceMatrix;

CovarianceMatrix* covariance_create( uint16_t );
CovarianceMatrix* covariance_create_empty( uint16_t );
CovarianceMatrix* covariance_create_zero( uint16_t );
void covariance_delete( CovarianceMatrix* );
void covariance_free_cache( CovarianceMatrix* );

int covariance_save( CovarianceMatrix*, FILE* );
CovarianceMatrix* covariance_load( FILE* );

CovarianceMatrix* covariance_marginalize( CovarianceMatrix*, uint16_t, uint16_t*);

CovarianceMatrix* covariance_use( CovarianceMatrix* );

void covariance_invalidate_cache( CovarianceMatrix* );

//uint8_t covariance_copy( CovarianceMatrix*, CovarianceMatrix*, uint8_t);

void covariance_set_zero( CovarianceMatrix* );
void covariance_set_identity( CovarianceMatrix* );
void covariance_set_full( CovarianceMatrix*, double* );
void covariance_set_diagonal_uniform( CovarianceMatrix*, double);
void covariance_set_diagonal( CovarianceMatrix*, double*);

void covariance_update_diag_scaling_factor( CovarianceMatrix* );
void covariance_alloc_LU( CovarianceMatrix* );
void covariance_update_LU( CovarianceMatrix* );
void covariance_alloc_icov( CovarianceMatrix* );
void covariance_update_icov( CovarianceMatrix* );
void covariance_alloc_ichol( CovarianceMatrix* );
void covariance_update_ichol( CovarianceMatrix* );
void covariance_alloc_sigma( CovarianceMatrix* );
void covariance_update_sigma( CovarianceMatrix* );
void covariance_update_cache( CovarianceMatrix* );

LU* covariance_get_lu( CovarianceMatrix* );
ICov* covariance_get_icov( CovarianceMatrix* );
IChol* covariance_get_ichol( CovarianceMatrix* );
SigmaPoints* covariance_get_sigmapoints( CovarianceMatrix* );

void covariance_print( CovarianceMatrix* );

#endif
