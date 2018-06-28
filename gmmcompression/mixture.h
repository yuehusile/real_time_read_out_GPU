#ifndef MIXTURE_H
#define MIXTURE_H

#include <stdio.h>
#include <stdlib.h>

#include "component.h"
#include "covariance.h"

#define DEFAULTBUFFERSIZE 100

void print_gsl_matrix( gsl_matrix* );

typedef struct mixture_t {
    uint32_t refcount;
    double sum_n;
    double sum_weights;
    uint16_t ndim;
    uint32_t ncomponents;
    uint32_t buffersize;
    GaussianComponent **components;
    CovarianceMatrix *samplecovariance;
} Mixture;

Mixture* mixture_create( uint16_t );
void mixture_delete( Mixture* );
void mixture_grow_buffer( Mixture*, uint32_t);
void mixture_shrink_buffer( Mixture* );
Mixture* mixture_use( Mixture* );

Mixture* mixture_marginalize( Mixture*, uint16_t, uint16_t*);

void mixture_remove_all_components( Mixture* );
uint32_t mixture_add_component( Mixture*, GaussianComponent* );
uint32_t mixture_add_multiple_components( Mixture*, GaussianComponent**, uint32_t);
void mixture_remove_component( Mixture*, GaussianComponent* );
void mixture_remove_components_by_index( Mixture*, uint32_t, uint32_t );
int32_t mixture_find_component( Mixture*, GaussianComponent* );
GaussianComponent* mixture_get_component( Mixture*, uint32_t );

void mixture_update_cache( Mixture* );
int mixture_save( Mixture*, FILE* );
Mixture* mixture_load( FILE* );
int mixture_save_to_file( Mixture* , const char* );
Mixture* mixture_load_from_file( const char* );
Mixture* copy_mixture( Mixture* );

void mixture_get_means( Mixture* , double* );
void mixture_get_scaling_factors( Mixture*, double* );
void mixture_get_weights( Mixture* , double* );
void mixture_get_covariances( Mixture*, double* );

void mixture_set_samplecovariance( Mixture*, CovarianceMatrix*);
CovarianceMatrix* mixture_get_samplecovariance( Mixture* );
void mixture_addsamples( Mixture*, double*, uint32_t, uint16_t );

void mixture_evaluate( Mixture*, double*, uint32_t, double*);
void mixture_evaluategrid( Mixture*, double*, uint32_t, uint16_t, uint16_t*, double*, uint32_t, uint16_t, uint16_t*, double* );
void mixture_evaluate_diagonal( Mixture*, double*, uint32_t, double*);
//void mixture_evaluategrid_diagonal( Mixture*, double*, uint32_t, uint16_t, uint16_t*, double*, uint32_t, uint16_t, uint16_t*, double* );

Mixture* mixture_compress( Mixture*, double, uint8_t );

void mixture_merge_samples( Mixture*, double*, uint32_t, double );
void mixture_merge_samples_match_bandwidth( Mixture*, double*, uint32_t, double );
void mixture_merge_samples_constant_covariance( Mixture*, double*, uint32_t, double );

typedef struct modelnode_t {
    struct modelnode_t *next;
    GaussianComponent** components;
    uint32_t ncomponents;
    GaussianComponent *model;
    double distance;
} ModelNode;

typedef struct modellist_t {
    ModelNode *head;
    uint32_t nmodels;
} ModelList;

void evaluate( GaussianComponent**, uint32_t, double*, uint32_t, double* );
void evaluate_diagonal( GaussianComponent**, uint32_t, double*, uint32_t, double* );

void mixture_evaluategrid_diagonal( Mixture*, double*, uint32_t, double*, uint16_t, uint16_t*, double* );
void mixture_prepare_grid_accumulator( Mixture* , double* , uint32_t , uint16_t , uint16_t* , double* );

void moment_match( GaussianComponent**, uint32_t, GaussianComponent*, uint8_t );
void moment_match_bandwidth( GaussianComponent**, uint32_t, GaussianComponent*, uint8_t );
GaussianComponent* modellist_pop( ModelList* , GaussianComponent*** , uint32_t* );
void modellist_add( ModelList* , GaussianComponent** , uint32_t, GaussianComponent* , double);
Mixture* model2mixture( ModelList*  ) ;
uint8_t assign_responsibilities( GaussianComponent**, uint32_t , GaussianComponent*, GaussianComponent*, GaussianComponent*** , GaussianComponent***, uint32_t*, uint32_t*);
double compute_distance( GaussianComponent*, GaussianComponent*);
double hellinger_single( GaussianComponent**, uint32_t, GaussianComponent*, uint8_t );

double compute_mahalanobis_distance( GaussianComponent*, double* );

double mixture_prepare_weights( Mixture*, uint32_t, double );

void mixture_evaluategrid_diagonal_multi( Mixture* mixture, double* grid_acc, uint32_t ngrid, double* points, uint32_t npoints, uint16_t npointsdim, uint16_t* pointsdim, double* output );

#endif /* mixture.h */
