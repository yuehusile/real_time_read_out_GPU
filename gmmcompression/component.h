#ifndef COMPONENT_H
#define COMPONENT_H

#include <inttypes.h>
#include "covariance.h"

/*
typedef struct sigma_points_cache {
    uint8_t is_outdated;
    double *data;
    uint16_t nsigmapoints;
} SigmaPointsCache;
*/

typedef struct gaussian_component {
    uint32_t refcount;
    uint16_t ndim;
    double weight;
    double *mean;
    CovarianceMatrix *covariance;
//    SigmaPointsCache sigmapoints;
} GaussianComponent;


GaussianComponent* component_create( uint16_t, double*, CovarianceMatrix*, double );
GaussianComponent* component_create_zero( uint16_t );
GaussianComponent* component_create_empty( uint16_t );
void component_delete( GaussianComponent* );

int component_save( GaussianComponent*, FILE* );
GaussianComponent* component_load( FILE* );

GaussianComponent* component_marginalize( GaussianComponent*, uint16_t, uint16_t*);

GaussianComponent* component_use( GaussianComponent* );

void component_set_zero( GaussianComponent* );

void component_set_weight( GaussianComponent*, double);
double component_get_weight( GaussianComponent* );

void component_set_mean( GaussianComponent*, double*);
void component_set_mean_uniform( GaussianComponent*, double);
void component_get_mean( GaussianComponent*, double* );

void component_get_covariance_array( GaussianComponent*, double* );

void component_set_covariance( GaussianComponent*, CovarianceMatrix*);
void component_set_covariance_zero ( GaussianComponent* );
void component_set_covariance_identity( GaussianComponent* );
void component_set_covariance_diagonal( GaussianComponent*, double*);
void component_set_covariance_diagonal_uniform( GaussianComponent*, double );
void component_set_covariance_full( GaussianComponent*, double* );
CovarianceMatrix* component_get_covariance( GaussianComponent* );

void component_split(GaussianComponent*, GaussianComponent**, GaussianComponent**);
void component_split_inplace(GaussianComponent*, GaussianComponent*, GaussianComponent*);
//component_update_sigmapoints( GaussianComponent* );
//void component_update_cache( GaussianComponent* );
void component_print( GaussianComponent* );

#endif
