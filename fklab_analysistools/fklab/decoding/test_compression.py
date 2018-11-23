
import numpy as np
import tables
import scipy as sp
import scipy.interpolate

import fklab.utilities.kernelsmoothing as ks
import fklab.containers.segments as seg
import fklab.containers.events as evt
import fklab.utilities.vectortools as vec
import fklab.decoding.kde as kde

import time

#load data
def load_dataset(path_to_data):
    F = tables.openFile(path_to_data)
    data = {'behavior':{}, 'mua':{}}

    obj = F.getNode("/behavior", "timestamp")
    behavior_time = obj.read().ravel()

    obj = F.getNode("/behavior", "xyposition")
    xy = np.ascontiguousarray( obj.read().T )

    obj = F.getNode("/behavior", "speed")
    speed = obj.read().ravel()

    spike_time = [obj_.read().ravel() for obj_ in F.walkNodes("/mua/timestamp", "Array")]
    spike_amp = [np.ascontiguousarray(obj_.read().T) for obj_ in F.walkNodes("/mua/amplitude", "Array")]
    spike_width = [obj_.read().ravel() for obj_ in F.walkNodes("/mua/spikewidth", "Array")]
    F.close()

    return behavior_time, xy, speed, spike_time, spike_amp, spike_width

behav_t, behav, _, spike_t, spike_amp, _ = load_dataset( '/home/fabian/Data/data.h5' )

#interpolate behavior if necessary (to remove NaNs)
invalid = np.isnan(np.sum(behav,axis=1))
behav[invalid] = sp.interpolate.interp1d( np.flatnonzero(~invalid), behav[~invalid],kind='linear',axis=0)( np.flatnonzero( invalid ) )

#define behavioral epoch
epoch = seg.Segment( [behav_t[0], behav_t[-1]] )

#remove spikes out of range
for k in range(len(spike_t)):
    idx = np.logical_and( spike_t[k]>=behav_t[0] , spike_t[k]<=behav_t[-1] )
    spike_t[k] = spike_t[k][idx]
    spike_amp[k] = spike_amp[k][idx]

#nearest neighbor interpolation of behavior at spike times
spike_behav = [ sp.interpolate.interp1d( behav_t, behav, kind='nearest', axis=0)( k ) for k in spike_t ]

#combine spike behavior and amplitude data
spike_data = [ np.column_stack( (x,y) ) for x,y in zip( spike_behav, spike_amp ) ]

#compute velocity, smooth
dt = np.mean(np.diff(behav_t))
velocity = np.gradient( behav[:,0] + 1j*behav[:,1], dt )
velocity = ks.Smoother( kernel=ks.GaussianKernel(bandwidth=0.25) )(velocity,delta=dt)
speed = np.abs(velocity)

#determine RUN and STOP segments
RUN = seg.Segment.fromlogical(speed>10,x=behav_t)
STOP = seg.Segment.fromlogical(speed<5,x=behav_t)

#split RUN in 1 second bins and partition into training/testing sets
RUN_train, RUN_test = RUN.split( size=1.0 ).partition( nparts=2, method='sequence', keepremainder=False )
RUN_test = RUN_test.split( size=0.25 )

#select behav and spikes in RUN_train [optionally filter on spike amplitude]
run_behav = behav[RUN_train.contains( behav_t )[0]]
run_spike = [ x[RUN_train.contains(t)[0]] for x,t in zip(spike_data,spike_t) ]



#-----------

#add samples in batches to density
def batch_compression( density, data, batchsize=25 ):
    
    from fklab.utilities.general import blocks
    
    n,ndim = data.shape
    
    assert( density.ndim==ndim)
    
    nsamples = []
    ncomponents = []
    
    for a,b in blocks( n, batchsize ):
        density.addsamples( data[a:(a+b)] )
        nsamples.append( b )
        ncomponents.append( density._mixture.ncomponents )
    
    return np.cumsum( nsamples ), np.array( ncomponents )

import time
import collections

decode_mode = '1D' #or '2D'

compression_methods = ['bandwidth',] #['full','bandwidth','constant']
if decode_mode == '1D':
    compression_thresholds = [2.0,] #[0.75,1.0,1.25,1.5,1.75,2.,2.5,3.0,3.5,4.,5.,6.]
    xgrid_vector = np.arange( behav[:,0].min(), behav[:,0].max(), 5 )
    grid = xgrid_vector.reshape( (len(xgrid_vector),1) ) #1D
    nx = len(xgrid_vector)
    true_behav = sp.interpolate.interp1d( behav_t, behav[:,0], kind='nearest', axis=0)( RUN_test.center ) #1D
else:
    compression_thresholds = [1.,1.5,2.0,2.5,3.0,4.,5.,6.]
    xgrid_vector = np.arange( behav[:,0].min(), behav[:,0].max(), 5 )
    ygrid_vector = np.arange( behav[:,1].min(), behav[:,1].max(), 5 )
    xgrid,ygrid = np.meshgrid( xgrid_vector, ygrid_vector, indexing='ij')
    grid = np.column_stack( (xgrid.ravel(), ygrid.ravel()) ) #2D
    nx = len(xgrid_vector)
    ny = len(ygrid_vector)
    true_behav = sp.interpolate.interp1d( behav_t, behav, kind='nearest', axis=0)( RUN_test.center ) #2D

result = collections.defaultdict( dict )
result['info']['thresholds'] = compression_thresholds
result['info']['methods'] = compression_methods
result['info']['description'] = decode_mode + ' decoding of open field data'
result['info']['ngrid'] = len(grid)
result['info']['training_time'] = np.sum( RUN_train.duration )
result['info']['testing_time'] = np.sum( RUN_test.duration )
result['info']['n_testing_bins'] = len( RUN_test )
result['info']['decode_bin_size'] = 0.25
result['info']['sample_covariance'] = 10.0
result['info']['sample_covariance_spike_amp'] = 0.03
result['info']['n_tetrodes'] = 9
result['info']['n_behav_samples'] = len(run_behav)
result['info']['n_train_spikes'] = [ len(x) for x in run_spike ]
result['info']['n_test_spikes'] = [ np.sum(RUN_test.contains(t)[0]) for x,t in zip(spike_data,spike_t) ]

for method in compression_methods:
    
    print("method = " + method)
    
    result[method]['behav_density'] = collections.defaultdict( dict )
    result[method]['spike_density'] = collections.defaultdict( dict )
    result[method]['decoder'] = collections.defaultdict( dict )
    
    result[method]['behav_density']['construction_time'] = []
    result[method]['behav_density']['construction_evolution'] = []
    result[method]['behav_density']['ncomponents'] = []
    
    result[method]['spike_density']['construction_time'] = []
    result[method]['spike_density']['construction_evolution'] = []
    result[method]['spike_density']['ncomponents'] = []
    
    result[method]['decoder']['construction_time'] = []
    result[method]['decoder']['decoding_time'] = []
    result[method]['decoder']['median_error'] = []
    
    for threshold in compression_thresholds:
        
        print("threshold = " + str(threshold) )
        
        #construct densities
        t = time.time()
        if decode_mode == '1D':
            Dbehav = kde.MergingCompressionDensity( ndim=1, sample_covariance = np.diag( [10.0] )**2, method = method, threshold = threshold ) #1D
            nsamples, ncomponents = batch_compression( Dbehav, run_behav[:,0:1], batchsize=25 ) #1D
        else:
            Dbehav = kde.MergingCompressionDensity( ndim=2, sample_covariance = np.diag( [10.0,10.0] )**2, method = method, threshold = threshold ) #2D
            nsamples, ncomponents = batch_compression( Dbehav, run_behav, batchsize=25 ) #2D
        
        t = time.time() - t
        
        result[method]['behav_density']['construction_time'].append( t )
        result[method]['behav_density']['construction_evolution'].append( (nsamples,ncomponents) )
        result[method]['behav_density']['ncomponents'].append( ncomponents[-1] )
        
        Dspike = []
        tmp_time = []
        tmp_evolution = []
        tmp_ncomp = []
        for x in run_spike:
            
            t = time.time()
            if decode_mode == '1D':
                density = kde.MergingCompressionDensity( ndim=5, sample_covariance = np.diag( [10.0,0.03,0.03,0.03,0.03] )**2, method = method, threshold = threshold ) #1D
                nsamples, ncomponents = batch_compression( density, x[:,[0,2,3,4,5]], batchsize=25 ) #1D
            else:
                density = kde.MergingCompressionDensity( ndim=6, sample_covariance = np.diag( [10.0,10.0,0.03,0.03,0.03,0.03] )**2, method = method, threshold = threshold ) #2D
                nsamples, ncomponents = batch_compression( density, x[:,[0,1,2,3,4,5]], batchsize=25 ) #2D
            Dspike.append(density)
            t = time.time() - t
            
            tmp_time.append(t)
            tmp_evolution.append( (nsamples, ncomponents) )
            tmp_ncomp.append( ncomponents[-1] )
            
            
        result[method]['spike_density']['construction_time'].append( tmp_time )
        result[method]['spike_density']['construction_evolution'].append( tmp_evolution )
        result[method]['spike_density']['ncomponents'].append( tmp_ncomp )
        
        print("densities constructed, starting decoding...")
        
        #construct decoder object
        t = time.time()
        decoder = kde.KDEDecoder( Dbehav, Dspike, np.sum( RUN_train.duration ), grid, offset=0.01 )
        t = time.time() - t
        
        result[method]['decoder']['construction_time'].append( t )
        
        t = time.time()
        posterior = decoder.decode2( spike_amp, spike_t, RUN_test, experimental=False )
        t = time.time() - t
        
        result[method]['decoder']['decoding_time'].append( t )
        
        if decode_mode == '1D':
            map_estimator = grid[ np.argmax( posterior, axis=0 ) ].ravel() #1D
            median_error = np.median( np.abs( map_estimator - true_behav ) ) #1D
        else:
            q = np.unravel_index( np.argmax( posterior, axis=0 ), (nx,ny) ) #2D
            err = np.sqrt( (xgrid_vector[ q[0] ]-true_behav[:,0])**2 + (ygrid_vector[ q[1] ]-true_behav[:,1])**2 ) #2D
            median_error = np.median( err ) #2D
        
        result[method]['decoder']['median_error'].append( median_error )


#plotting
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_context('paper')

save_path = '/home/fabian/Data/AnalysisCompressionDecoding/' + decode_mode
dpi = 300

if decode_mode == '1D':
    low,medium,high = 0,5,9 #indices for low, medium and high thresholds
else:
    low,medium,high = 0,2,5

#plot pi(x) compression evolution: #components vs #samples 
nrows,ncols=2,len(compression_methods)
hF, hAx = plt.subplots( nrows=nrows, ncols=ncols, sharex=True, sharey='row', squeeze=False )
plt.subplots_adjust( bottom=0.2 )

pal = sns.color_palette()

hF.set_size_inches([ncols*2.5,nrows*2.5],forward=True )

plt.suptitle( 'Evolution of pi(x) compression' )

for colidx,method in enumerate(compression_methods):
    plt.axes( hAx[0,colidx] )
    hLines = [ [plt.plot( x1, x2, color=pal[k] ) for x1,x2 in [result[method]['behav_density']['construction_evolution'][idx]] ] for k,idx in enumerate([low,medium,high]) ]
    _ = [x[0][0].set_label( 'threshold = ' + str(compression_thresholds[y]) ) for x,y in zip( hLines, [low,medium,high] )]
    plt.legend( loc='upper left', fontsize='xx-small' )
    plt.ylabel( 'number of components' )
    plt.title('method = ' + method)
    plt.yscale('log')
    plt.xscale('log')
    
    plt.axes( hAx[1,colidx] )
    hLines = [ [plt.plot( x1, x1/x2, color=pal[k] ) for x1,x2 in [result[method]['behav_density']['construction_evolution'][idx]] ] for k,idx in enumerate([low,medium,high]) ]
    _ = [x[0][0].set_label( 'threshold = ' + str(compression_thresholds[y]) ) for x,y in zip( hLines, [low,medium,high] )]
    plt.legend( loc='upper left', fontsize='xx-small' )
    plt.xlabel( 'number of samples' )
    plt.ylabel( 'compression factor' )
    plt.yscale('log')
    plt.xscale('log')

plt.savefig( os.path.join( save_path, 'Evolution_pix.png'), dpi=dpi )


#plot p(a,x) compression evolution: #components vs #samples for all tetrodes
nrows,ncols = 2,len(compression_methods)
hF, hAx = plt.subplots( nrows=nrows, ncols=ncols, sharex=True, sharey='row', squeeze=False )
plt.subplots_adjust( bottom=0.2 )

pal = sns.color_palette()

hF.set_size_inches([ncols*2.5,nrows*2.5],forward=True )

plt.suptitle( 'Evolution of p(a,x) compression for all tetrodes' )

for colidx,method in enumerate(compression_methods):
    plt.axes( hAx[0,colidx] )
    hLines = [ [plt.plot( x[0], x[1], color=pal[k] ) for x in result[method]['spike_density']['construction_evolution'][idx] ] for k,idx in enumerate([low,medium,high]) ]
    _ = [x[0][0].set_label( 'threshold = ' + str(compression_thresholds[y]) ) for x,y in zip( hLines, [low,medium,high] )]
    plt.legend( loc='upper left', fontsize='xx-small' )
    plt.ylabel( 'number of components' )
    plt.title('method = ' + method)
    plt.yscale('log')
    plt.xscale('log')
    
    plt.axes( hAx[1,colidx] )
    hLines = [ [plt.plot( x[0], x[0]/x[1], color=pal[k] ) for x in result[method]['spike_density']['construction_evolution'][idx] ] for k,idx in enumerate([low,medium,high]) ]
    _ = [x[0][0].set_label( 'threshold = ' + str(compression_thresholds[y]) ) for x,y in zip( hLines, [low,medium,high] )]
    plt.legend( loc='upper left', fontsize='xx-small' )
    plt.xlabel( 'number of samples' )
    plt.ylabel( 'compression factor' )
    plt.yscale('log')
    plt.xscale('log')

plt.savefig( os.path.join( save_path, 'Evolution_pax.png'), dpi=dpi )


#plot compression of pi(x) as function of threshold
thresholds = result['info']['thresholds']
nsamples = np.array(result['info']['n_behav_samples'])

nrows,ncols=2,len(compression_methods)

hF, hAx = plt.subplots( nrows=nrows, ncols=ncols, sharex=True, sharey='row', squeeze=False )
plt.subplots_adjust( bottom=0.2 )

pal = sns.color_palette()

hF.set_size_inches([ncols*2.5,nrows*2.5],forward=True )

plt.suptitle( 'Compression of pi(x)' )

for colidx,method in enumerate(compression_methods):
    plt.axes( hAx[0,colidx] )
    hLines = plt.plot( thresholds, np.array(result[method]['behav_density']['ncomponents']), color=pal[colidx] )
    plt.ylabel( 'number of components' )
    plt.title( 'method = ' + method)
    plt.yscale('log')
    
    plt.axes( hAx[1,colidx] )
    hLines = plt.plot( thresholds, nsamples / np.array(result[method]['behav_density']['ncomponents']), color=pal[colidx] )
    plt.ylabel( 'compression factor' )
    plt.xlabel( 'threshold' )
    plt.yscale('log')

plt.savefig( os.path.join( save_path, 'Compression_pix.png'), dpi=dpi )


#plot compression of p(a,x) as function of threshold
thresholds = result['info']['thresholds']
nsamples = np.array(result['info']['n_train_spikes'])

nrows,ncols=2,len(compression_methods)
hF, hAx = plt.subplots( nrows=nrows, ncols=ncols, sharex=True, sharey='row', squeeze=False )
plt.subplots_adjust( bottom=0.2)

pal = sns.color_palette()

hF.set_size_inches([ncols*2.5,nrows*2.5],forward=True )

plt.suptitle( 'Compression of p(a,x)' )

for colidx,method in enumerate(compression_methods):
    plt.axes( hAx[0,colidx] )
    hLines = plt.plot( thresholds, np.array(result[method]['spike_density']['ncomponents']), color=pal[colidx] )
    plt.ylabel( 'number of components' )
    plt.title( 'method = ' + method)
    plt.yscale('log')
    
    plt.axes( hAx[1,colidx] )
    hLines = plt.plot( thresholds, nsamples / np.array(result[method]['spike_density']['ncomponents']), color=pal[colidx] )
    plt.ylabel( 'compression factor' )
    plt.xlabel( 'threshold' )
    plt.yscale('log')

plt.savefig( os.path.join( save_path, 'Compression_pax.png'), dpi=dpi )


#plot decoding error
thresholds = result['info']['thresholds']

nrows,ncols=1,2
hF, hAx = plt.subplots( nrows=nrows, ncols=ncols, sharex=False, sharey='row', squeeze=False )
plt.subplots_adjust( bottom=0.2 )

pal = sns.color_palette()

hF.set_size_inches([ncols*2.5,nrows*2.5],forward=True )

plt.suptitle('Decoding error' )

for colidx,method in enumerate(compression_methods):
    
    plt.axes( hAx[0,0] )
    plt.plot( thresholds, result[method]['decoder']['median_error'], color=pal[colidx], label = 'method = ' + method );
    
    plt.axes( hAx[0,1] )
    plt.plot( np.mean( np.array(result[method]['spike_density']['ncomponents']), axis=1 ), result[method]['decoder']['median_error'], color=pal[colidx], label = 'method = ' + method );

plt.axes( hAx[0,0] )
plt.legend( loc='upper left', fontsize='xx-small' )
plt.ylabel('median error (cm)')
plt.xlabel('threshold')
#plt.ylim( [0,plt.ylim()[1]] )

plt.axes( hAx[0,1] )
plt.legend( loc='upper right', fontsize='xx-small' )
plt.ylabel('median error (cm)')
plt.xlabel('mean number\nof components')
hAx[0,1].xaxis.get_major_formatter().set_powerlimits((-3,3))

plt.savefig( os.path.join( save_path, 'Decoding_error.png'), dpi=dpi )


#plot compression time
thresholds = result['info']['thresholds']
nsamples = np.sum( np.array(result['info']['n_train_spikes']) )

nrows,ncols=1,2
hF, hAx = plt.subplots( nrows=nrows, ncols=ncols, sharex=False, sharey='row', squeeze=False )
plt.subplots_adjust( bottom=0.2 )

pal = sns.color_palette()

hF.set_size_inches([ncols*2.5,nrows*3],forward=True )

plt.suptitle('Compression time' )


for colidx,method in enumerate(compression_methods):
    
    plt.axes( hAx[0,0] )
    plt.plot( thresholds, 1000*np.sum( np.array(result[method]['spike_density']['construction_time']), axis=1 ) / nsamples, color=pal[colidx], label = 'method = ' + method );
    
    plt.axes( hAx[0,1] )
    plt.plot( np.mean( np.array(result[method]['spike_density']['ncomponents']), axis=1 ), 1000*np.sum( np.array(result[method]['spike_density']['construction_time']), axis=1 ) / nsamples, color=pal[colidx], label = 'method = ' + method );

plt.axes( hAx[0,0] )
plt.legend( loc='upper right', fontsize='xx-small' )
plt.ylabel('mean time/sample (ms)')
plt.xlabel('threshold')

plt.axes( hAx[0,1] )
plt.legend( loc='upper left', fontsize='xx-small' )
plt.ylabel('mean time/sample (ms)')
plt.xlabel('mean number\nof components')
hAx[0,1].xaxis.get_major_formatter().set_powerlimits((-3,3))

plt.savefig( os.path.join( save_path, 'Compression_time.png'), dpi=dpi )


#plot decoding time
thresholds = result['info']['thresholds']
nsamples = np.sum( np.array(result['info']['n_test_spikes']) )

nrows,ncols=1,2
hF, hAx = plt.subplots( nrows=1, ncols=2, sharex=False, sharey='row', squeeze=False )
plt.subplots_adjust( bottom=0.2 )

pal = sns.color_palette()

hF.set_size_inches([ncols*2.5,nrows*2.5],forward=True )

plt.suptitle('Decoding time' )


for colidx,method in enumerate(compression_methods):
    
    plt.axes( hAx[0,0] )
    plt.plot( thresholds, 1000*np.array(result[method]['decoder']['decoding_time']) / nsamples, color=pal[colidx], label = 'method = ' + method );
    
    plt.axes( hAx[0,1] )
    plt.plot( np.mean( np.array(result[method]['spike_density']['ncomponents']), axis=1 ), 1000*np.array(result[method]['decoder']['decoding_time']) / nsamples, color=pal[colidx], label = 'method = ' + method );


plt.axes( hAx[0,0] )
plt.legend( loc='upper right', fontsize='xx-small' )
plt.ylabel('mean time/sample (ms)')
plt.xlabel('threshold')

plt.axes( hAx[0,1] )
plt.legend( loc='upper left', fontsize='xx-small' )
plt.ylabel('mean time/sample (ms)')
plt.xlabel('mean number\nof components')
hAx[0,1].xaxis.get_major_formatter().set_powerlimits((-3,3))

plt.savefig( os.path.join( save_path, 'Decoding time.png'), dpi=dpi )
