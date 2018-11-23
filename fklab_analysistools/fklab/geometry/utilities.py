"""
============================================================
Geometry utility functions (:mod:`fklab.geometry.utilities`)
============================================================

.. currentmodule:: fklab.geometry.utilities

Utilities for conversion between cartesian and polar coordinates,
for working with polygons, for projecting points to (poly)lines and for 
computing shortest path in graphs.

.. autosummary::
    :toctree: generated/
    
    cart2pol
    pol2cart
    inpoly
    polygonarea
    point2line
    point2polyline
    aspoints
    floyd_warshall
    shortest_path
    
"""

import numpy as np
from numba import jit, int_

__all__ = ['cart2pol','pol2cart','inpoly','polygonarea','point2line',
           'point2polyline', 'aspoints', 'floyd_warshall',
           'shortest_path']

def cart2pol(x,y):
    """Converts cartesian to polar coordinates.
    
    Parameters
    ----------
    x,y  : ndarray
    
    Returns
    -------
    theta, rho
    
    """
    theta = np.arctan2(y, x)
    rho = np.sqrt(x**2 + y**2)
    return (theta, rho)

def pol2cart(theta,rho):
    """Converts polar to cartesian coordinates.
    
    Parameters
    ----------
    theta, rho  : ndarray
    
    Returns
    -------
    x, y
    
    """
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return (x,y)

@jit(locals={'npoints':int_,'nvertex':int_,'i':int_,'j':int_}, nopython=True, nogil=True)
def inpoly(nodes,xy):
    """Tests if points are inside polygon.
    
    Parameters
    ----------
    nodes : (p,2) array
        Coordinates of polygon nodes
    xy : (n,2) array
        Coordinates of points to test
    
    Returns
    -------
    (n,) array
    
    """
    npoints = xy.shape[0]
    nvertex = nodes.shape[0]
    
    b = np.zeros(npoints)
    
    for i in range(nvertex-1):
        for j in range(npoints):
            if nodes[i,1]<=xy[j,1] and nodes[i+1,1]>xy[j,1]: #upward crossing
                if ((nodes[i+1,0]-nodes[i,0])*(xy[j,1]-nodes[i,1]) - (xy[j,0]-nodes[i,0])*(nodes[i+1,1]-nodes[i,1]))>0 :
                    b[j]+=1
            elif nodes[i,1]>xy[j,1] and nodes[i+1,1]<=xy[j,1]: #downward crossing
                if ((nodes[i+1,0]-nodes[i,0])*(xy[j,1]-nodes[i,1]) - (xy[j,0]-nodes[i,0])*(nodes[i+1,1]-nodes[i,1]))<0 :
                    b[j]-=1
    return b!=0

def polygonarea(nodes):
    """Computes area of polygon.
    
    Parameters
    ----------
    nodes: (p,2) array
        Coordinates of polygon nodes
    
    Returns
    -------
    scalar
    
    """
    #TODO: make sure nodes is a (n,2) array
    n = nodes.shape[0]
    if n<=2:
        return 0
    
    if ~np.all( nodes[0]==nodes[-1] ):
        nodes = np.concatenate( (nodes, nodes[0:1]), axis=0 )
    
    return 0.5* np.sum( nodes[0:-1,0] * nodes[1:,1] - nodes[1:,0] * nodes[0:-1,1] )
        
def point2line(line,points,clip=('normal','normal')):
    """Compute distance between point and line.
    
    Parameters
    ----------
    line: (2,2) array_like
        Two (x,y) pairs that define the end points of a line segment
    points: (n,2) array_like
        Array of (x,y) pairs - the points to be tested
    clip: 2-element tuple
        Clipping behavior for the two line segment end points.
        'normal' = points beyond line segment are projected to the line end points
        'blunt' = points beyond line segment are excluded
        'full' = points are projected to the full line that passes through the two end points
    
    Returns
    -------
    distance: (n,) vector
        Distance from each point to the line segment (or nan if clip='blunt'
        and the point lies beyond the line segment). The sign of the distance
        reflects on which side of the line the point is located. 
    points: (n,2) array
        For each input point, this array contains the nearest point on the line
    
    """
    #TODO: make sure `line` is a (2,2) array
    #TODO: make sure `points` is a (n,2) array
    #TODO: make sure clip is a string or 2-element tuple
    
    #number of points to test
    npoints = points.shape[0]
    
    #compute line vector as difference between the two end points
    v = np.diff(line,1,axis=0)
    #express points relative to first line end point
    w = points - line[0:1,:]
    
    #project points to line through end points
    c1 = np.dot( w, v.T ).ravel()
    c2 = np.dot( v, v.T ).ravel() #squared length of line vector

    b = c1 / c2
    
    #compute projection points on the full line
    P = line[0:1,:] + v[0:1,:]*b[:,np.newaxis]
    #compute distance point and projected point
    D = np.sqrt( np.sum( (P - points)**2, axis=1 ) )
    #compute linear distance along line (from first line end point)
    LD = b * np.sqrt(c2) 

    #determine sign of distance (i.e. if points are located on one or
    #the other side of the line)
    #imagine the coordinate system is rotated such that the line is on
    #the x-axis, with the line vector pointing to positive
    #then points below the x-axis have negative distance
    
    dpoint = points - P
    ss = np.sign( np.sin( np.arctan2( dpoint[:,1], dpoint[:,0] ) - np.arctan2( v[0,1], v[0,0] ) ) )
    
    #deal with clipping at first line end point
    if clip[0] == 'normal':
        #find all points that are beyond first line end point
        idx = c1<=0
        n = np.sum(idx)
        if n>0:
            #compute distance to end point
            D[idx] = np.sqrt( np.sum( w[idx,:]**2, axis=1 ) )
            #set projected point to end point
            P[idx] = line[0,:]
            #set linear distance to 0
            LD[idx] = 0
    elif clip[0] == 'blunt':
        #exclude all points that are beyond first line end point
        idx = c1<0
        n = np.sum(idx)
        if n>0:
            D[idx] = np.nan
            P[idx] = np.nan
            LD[idx] = np.nan
            
    
    #deal with clipping at second line end point
    if clip[1] == 'normal':
        #find all points that are beyond second line end point
        idx = c2<=c1
        n = np.sum(idx)
        if n>0:
            #compute distance to end point
            D[idx] = np.sqrt( np.sum( (points[idx,:]-line[1,:])**2, axis=1 ) )
            #set projected point to end point
            P[idx] = line[1,:]
            #set linear distance to length of line
            LD[idx] = np.sqrt(c2)
    elif clip[1] == 'blunt':
        #exclude all points that are beyond second line end point
        idx = c2<c1
        n = np.sum(idx)
        if n>0:
            D[idx] = np.nan
            P[idx] = np.nan
            LD[idx] = np.nan            
    
    #correct sign of distance
    D = D * ss
    
    return (D,P,LD)

def point2polyline(nodes,points,clip=('normal','normal')):
    """Project points to polyline.
    
    Parameters
    ----------
    nodes : (p,2) array
        Coordinates of polygon nodes
    points : (n,2) array
        Coordinates of points to be projected to polyline
    clip : 2-element tuple
        Clipping behavior for the two polyline end points.
        'normal' = points beyond line segment are projected to the line end points
        'blunt' = points beyond line segment are excluded
        'full' = points are projected to the full line that passes through the end points
    
    Returns
    -------
    distance: (n,) vector
        Distance from each point to the polyline (or nan if clip='blunt'
        and the point lies beyond the line segment). The sign of the distance
        reflects on which side of the polyline the point is located. 
    points: (n,2) array
        For each input point, this array contains the nearest point on the line
    edge : (n,) vector
        The index of the polyline segment that each input point is closest to.
    edge_distance : (n,) vector 
        Distance to projected points from the start of the edge
    path_distance : (n,) vector
        Distance to projected points from the start of the polyline
    
    """
    
    #determine number of edges in polyline
    nedges = nodes.shape[0]
    #determine number of points to project
    npoints = points.shape[0]
    
    #test if we are working with a closed polyline
    isclosed = np.all(nodes[0,:]==nodes[-1,:])
    
    #setup output parameters
    dist_to_path = np.full(npoints,np.inf,dtype=np.float64)
    point_on_path = np.full((npoints,2),np.nan,dtype=np.float64)
    edge = np.full(npoints,-1,dtype=np.int)
    dist_along_edge = np.full(npoints,np.nan,dtype=np.float64)
    dist_along_path = np.full(npoints,np.nan,dtype=np.float64)
    
    #loop through all segments of polyline
    for s in range(nedges-1):
        
        #compute distance from points to segment
        #take care of start and end points
        if s==0 and not isclosed:
            [d,p,ld] = point2line(nodes[s:s+2], points, (clip[0], 'normal'))
        elif s==nedges-1 and not isclosed:
            [d,p,ld] = point2line(nodes[s:s+2], points, ('normal', clip[1]))
            #linear distance is defined in interval [0,L>, where L is
            #the total length of the polyline. Since L itself is not included,
            #we subtract a small amount for those points that are coincident
            #with the polyline end point 
            ld = np.minimum( ld, np.sqrt(np.sum(nodes[s:s+2]**2)) - np.spacing(1) )
        else:
            [d,p,ld] = point2line(nodes[s:s+2], points, ('normal','normal'))
        
        #retain smallest distance
        idx = np.abs(d)<np.abs(dist_to_path)
        n = np.sum(idx)
        if n>0:
            dist_to_path[idx] = d[idx]
            point_on_path[idx] = p[idx]
            edge[idx] = s
            dist_along_edge[idx] = ld[idx]
    
    #compute linear distance from the start of the polyline
    #by adding the cumulative lengths of edges preceding the edge that 
    #the point is projected to
    cumdist = np.cumsum( np.sqrt( np.sum( np.diff( nodes, 1, axis=0 )**2, axis=1 ) ) )
    cumdist = np.concatenate( ([0],cumdist) )
    
    valid = edge>=0
    dist_along_path[valid] = dist_along_edge[valid] + cumdist[ edge[valid] ]    
    dist_to_path[~valid] = np.nan
    
    return (dist_to_path, point_on_path, edge, dist_along_edge, dist_along_path)

def aspoints(x,ndim=2,copy=True):
    """Ensures array has shape (n,ndim).
    
    Parameters
    ----------
    x : ndarray
    ndim : scalar
    copy : bool
    
    Returns
    -------
    (n,ndim) array
    
    """
    
    x = np.array(x, copy=copy, ndmin=2)
    
    ndim = int(ndim)
    
    if x.size==0 or x.size==ndim:
        x = x.reshape( (x.size/ndim,ndim) )
    
    if x.ndim>2:
        raise ValueError("Point array should be 2-dimensional.")
    
    if x.shape[1]!=ndim:
        raise ValueError("Incorrect dimensionality of point array.")
    
    return x
    
def floyd_warshall(m):
    """All pair shortest path algorithm for weighted directed graphs.
    
    Parameters
    ----------
    m : 2d array
        square matrix where each (i,j) element contains the weigth or
        distance associated with the edge between nodes i and j. If no 
        edge exists, the element should be set to Inf. The elements on
        the diagonal should be set to zero.
    
    Returns
    -------
    m : 2d array
        square matrix with shortest distance between any pair of nodes.
    shortest_path(i,j) : function
        function that returns all the nodes that make up the shortest
        path between nodes i and j.
    
    """
    
    m = m.copy()
    
    r,c = m.shape
    nxt = np.full( (r,r), -1, dtype=np.int)
    
    m[m==0] = np.Inf #set zero entries to Inf
    m[np.diag_indices_from(m)] = 0 #set diagonal to zero
    
    for k in range(r):
        tmp = m[:,k:k+1] + m[k:k+1,:]
        mask = tmp < m
        m[mask] = tmp[mask]
        nxt[mask] = k
    
    fcn = lambda i,j: _shortest_path( m, nxt, i, j )
    
    return m, fcn

def shortest_path( m, i, j ):
    """Shortest path between two nodes in a graph.
    
    Parameters
    ----------
    m : 2d array
        square matrix where each (i,j) element contains the weigth or
        distance associated with the edge between nodes i and j. If no 
        edge exists, the element should be set to Inf. The elements on
        the diagonal should be set to zero.
    i,j : int
        node indices
    
    Returns
    -------
    distance: scalar
        shortest path distance between nodes
    path : list or None
        sequences of nodes in shortest path, or None if nodes are not
        connected
    
    """
    
    d, fcn = floyd_warshall( m )
    
    return d[i,j], fcn(i,j)

def _shortest_path( dist, nextnode, i, j ):
    
    if i==j:
        p = [i,]
    elif np.isinf( dist[i,j] ):
        p = None
    else:
        p = _shortest_path_recurse( nextnode, i, j )
        p = [i,] + p + [j,]
    
    return p

def _shortest_path_recurse( nextnode, i, j ):
    
    p = []
    intermediate = nextnode[ i, j ]
    
    if intermediate>=0:
        p = _shortest_path_recurse(nextnode, i, intermediate) + [intermediate,] + _shortest_path_recurse(nextnode, intermediate,j)
    
    return p
