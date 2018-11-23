
import numpy as np
import tables
import scipy as sp
import scipy.interpolate

import fklab.utilities.kernelsmoothing as ks
import fklab.containers.segments as seg
import fklab.containers.events as evt
import fklab.utilities.vectortools as vec
import fklab.decoding.kde as mixedkde

import time

#load data
def load_dataset(path_to_data):
    F = tables.openFile(path_to_data)
    data = {'behavior':{}, 'mua':{}}

    obj = F.getNode("/behavior", "timestamp")
    data['behavior']['timestamp'] = obj.read()

    obj = F.getNode("/behavior", "xyposition")
    data['behavior']['xyposition'] = obj.read()

    obj = F.getNode("/behavior", "speed")
    data['behavior']['speed'] = obj.read()

    data['mua']['timestamp'] = np.array([obj_.read() for obj_ in F.walkNodes("/mua/timestamp", "Array")])
    data['mua']['amplitude'] = np.array([obj_.read() for obj_ in F.walkNodes("/mua/amplitude", "Array")])
    data['mua']['spikewidth'] = np.array([obj_.read() for obj_ in F.walkNodes("/mua/spikewidth", "Array")])
    F.close()

    behavior = np.vstack((data['behavior']['xyposition'], data['behavior']['timestamp']))
    amplitude = [np.vstack((np.vstack(ch), np.vstack(t))) for ch, t in zip(data['mua']['amplitude'], data['mua']['timestamp'])]
    return amplitude, behavior

spike_amp, behav = load_dataset( '/home/fabian/Data/data.h5' )

#interpolate behavior if necessary (to remove NaNs)
invalid = np.isnan(np.sum(behav[0:2,:],axis=0))
behav[0:2,invalid] = sp.interpolate.interp1d( (~invalid).nonzero()[0], behav[0:2,~invalid],kind='linear',axis=1)(invalid.nonzero()[0] )

epoch = seg.Segment( [behav[2,0],behav[2,-1]] )

#remove spikes out of range
spike_amp = [ k[:, np.logical_and( k[4]>=behav[2,0] , k[4]<=behav[2,-1] )] for k in spike_amp ]
#nearest neighbor interpolation of behavior at spike times
spike_behav = [sp.interpolate.interp1d( behav[2],behav[0:2],kind='nearest',axis=1)(k[4]) for k in spike_amp]

spike_data = [ np.vstack( (x,y) ) for x,y in zip(spike_behav,spike_amp) ]

#compute velocity, smooth
dt = np.mean(np.diff(behav[2]))
velocity = np.gradient( behav[0] + 1j*behav[1], dt )
velocity = ks.Smoother( kernel=ks.GaussianKernel(bandwidth=0.25) )(velocity,delta=dt)
speed = np.abs(velocity)

#determine RUN and STOP segments
RUN = seg.Segment.fromlogical(speed>10,x=behav[2])
STOP = seg.Segment.fromlogical(speed<5,x=behav[2])

#select behav and spikes in run [optionally filter on spike amplitude]
run_behav = behav[:,RUN.contains( behav[2,:] )[0]]
run_spike = [ x[:,RUN.contains(x[6,:])[0]] for x in spike_data ]

#setup kde
#kde_behav = mixedkde.MixedKDE( np.ascontiguousarray( run_behav[0:2] ), bandwidths=10 )
#kde_spikes = [ mixedkde.MixedKDE( np.ascontiguousarray( k[0:6] ), bandwidths=[10,10,0.030,0.030,0.030,0.030] ) for k in run_spike ]

##gmm_behav = kc.compute_compressed_model( run_behav[0:2].T, covariance=np.diag( [10.0,10.0] )**2, samplesize=200, costThreshold=0.01 )
##gmm_spikes = [ kc.compute_compressed_model( k[0:6].T, covariance=np.diag( [10.0,10.0,0.03,0.03,0.03,0.03] )**2, samplesize=200, costThreshold=0.01 ) for k in run_spike ]

#define grid
xgrid_vector = np.arange( behav[0].min(), behav[0].max(), 5 )
ygrid_vector = np.arange( behav[1].min(), behav[1].max(), 5 )
xgrid,ygrid = np.meshgrid( xgrid_vector, ygrid_vector, indexing='ij')

grid = np.ascontiguousarray( np.vstack( (xgrid.ravel(), ygrid.ravel()) ) )

nx = len(xgrid_vector)
ny = len(ygrid_vector)

#compute PI(grid) kde
pix = kde_behav.evaluate( grid )
## pix = gmm_behav.evaluate( grid )

#for each tetrode compute P(grid) kde
px = [ k.evaluate_marginal( grid, dim=[0,1] ) for k in kde_spikes ]
## px = [ k.evaluate_marginal( grid, dim=[0,1] ) for k in gmm_spikes ]

#compile mua across tetrodes
#determine mean/std of mua in STOP segments
#detect mua bursts in STOP segments
timebins = epoch.split(size=0.001)
mua = evt.Event()
for k in spike_data:
    mua += evt.Event(k[6,:])
mua.sort()
mua_count = mua.bin( timebins )
mua_count_smooth = ks.Smoother( kernel=ks.GaussianKernel(bandwidth=0.015) )(mua_count.ravel(),delta=0.001)

mua_mean, mua_std = STOP.applyfcn( timebins.center, mua_count_smooth, function=lambda x: (np.mean(x),np.std(x)) )
bursts = vec.detect_mountains(mua_count_smooth, timebins.center, low=mua_mean, high=mua_mean+3*mua_std, segments=STOP)
bursts = bursts[bursts.duration>=0.1]

#split each burst into 20ms bins
burst_bins = bursts.split(size=0.02,join=False)
N = np.array([ len(k) for k in burst_bins ])
burst_bins = seg.Segment( seg.segment_concatenate(*burst_bins) )

#rate offset parameter
offset = 0.001

#compute mean rate for each tetrode
mu = [ k.shape[1]/np.sum(RUN.duration) for k in run_spike ]
#compute lambda(x) for each tetrode
lx = [ m*k/pix + offset for m,k in zip(mu,px)]

t1 = time.time()
nevents = 25 #number of events to decode
nbins = np.sum( N[0:nevents] )
cc = [0]*9 #for each tetrode an array with the number of spikes/bin
pax = 0
#loop over tetrodes
for z,(k,s) in enumerate(zip(kde_spikes,spike_data)):
    #find indices of all spikes inside bins and #spikes/bin
    bb,cc[z] = burst_bins[0:nbins].contains(s[-1])[0:2]
    bb = np.flatnonzero(bb)
    #evaluate kernel density
    tmp = k.evaluate_grid(grid,np.ascontiguousarray( s[np.ix_([2,3,4,5],bb)] ) )
    #transform to log, add offset
    tmp = np.log( tmp + offset*pix/mu[z] )
    #split bins
    tmp = np.split( tmp, np.cumsum(cc[z][0:-1]) )
    #sum over spikes in bins and subtract pi(x)
    tmp = [np.sum( qq, axis=0 ) - ccc*np.log(pix) if len(q)>0 else np.zeros(1,nx*ny) for qq,ccc in zip(tmp,cc[z])]
    tmp = np.vstack(tmp)
    pax += tmp

#subtract sum( p(x) )
P = pax - np.nansum( np.vstack(lx), axis=0 )*0.02
#normalize
P = P - np.nanmax(P,axis=1,keepdims=True)
P = np.exp(P)
P = P/np.nansum(P,axis=1,keepdims=True)

P = P.reshape(nbins,nx,ny)

#split events
P = np.split( P, np.cumsum( N[0:nevents-1] ) )

t2 = time.time()
print t2-t1


#test code for sparsifying distributions
#data should be normalized by standard deviation of desired kernel and threshold is expressed in terms of the standard deviation
def sparsify( data, threshold=0.5 ):
    
    threshold = threshold**2
    
    nd,nn = data.shape
    tmp = np.zeros( (nd,10000), dtype=np.float )
    w = np.zeros( 10000 )
    
    N=1
    tmp[:,0] = data[:,0]
    
    for k in xrange(1,nn):
        dd = tmp[:,:nn] - data[:,k:k+1]
        dd = np.sum(dd**2,axis=0)
        idx = np.argmin(dd)
        if dd[idx]<threshold:
            tmp[:,idx] = (tmp[:,idx]*w[idx] + data[:,k])/(w[idx]+1)
            w[idx]+=1
        elif N<10000:
            tmp[:,N] = data[:,k]
            w[N]=1
            N+=1
        else:
            raise ValueError
    
    return tmp[:,:N], w[:N]


