import math
import numpy as np
import scipy as sp
import scipy.spatial.distance as distance
import sys
from math import exp,sqrt
from numpy.random import choice,randint,seed
from scipy.spatial.distance import cdist,pdist,squareform

class density_peaks(object):
    """
    Rodriguez, A.; Laio, A. 
    Clustering by Fast Search and Find of Density Peaks. 
    Science 2014, 344 (6191), 1492–1496. 
    https://doi.org/10.1126/science.1242072.
    """
    def __init__(self,**kwargs):
        prop_defaults = {
            "metric"  : "euclidean",
            "kernel"  : "gaussian",
            "cutoff"  : 0.0,
            "percent" : 2.0,
        }
        for (prop, default) in prop_defaults.items():
            setattr(self, prop, kwargs.get(prop, default))
        kernels = ["flat","gaussian"]
        if self.kernel not in kernels:
            raise NotImplementedError("no kernel %s %s in" \
            % (self.kernel,self.__class__.__name__))
        if self.cutoff == 0.0:
            raise ValueError("cutoff must > 0")
        elif self.cutoff=="auto":
            print("Determining cutoff using a % of neighbors=",self.percent)

    def use_percent(self):
        """
        calculate average distance so that
        average number of neighbors within
        is percent of total points
        """
        dd = np.sort(self.D.ravel())
        N = dd.shape[0]
        frac = N*self.percent/100. 
        cutoff = dd[int(frac)]
        return cutoff

    def calc_rho(self):
        """
        calculate local density
        """
        rho = np.zeros(self.N)
        for i in range(self.N):
            mydist = self.D[i,:]
            if self.kernel == "flat":
                dens = float(len(mydist[mydist<self.cutoff]))-1.
            elif self.kernel == "gaussian":
                dens = np.sum(np.exp( -(mydist/self.cutoff)**2 ))-1
            rho[i] = dens
        rank = np.argsort(rho)[::-1]
        return rho, rank

    def calc_delta(self):
        """
        loop on ordered densities; for each point find
        minimum distance from other points with higher density
        the point with the highest density is assigned as 
        neighbor to build clusters
        """
        maxd = np.max(self.D)
        nneigh = np.zeros(self.N,dtype=np.int64)
        delta  = maxd*np.ones(self.N)
        delta[self.order[0]] = -1.
        for i in range(1,self.N):
            I = self.order[i]
            Imin = self.order[:i]
            dd = np.min(self.D[I,Imin])
            delta[I] = dd
            nn = self.order[np.where(self.D[I,Imin]==dd)[0][0]]
            nneigh[I] = nn
        delta[self.order[0]] = np.max(delta)
        #delta[self.order[0]] = np.max(self.D[self.order[0]])
        return delta,nneigh

    def decision_graph(self, X=None, D=None):
        """
        calculate decision graph for data set
        """
        # check X and/or D
        if self.metric=="precomputed" and D is None:
            raise ValueError("missing precomputed distance matrix")
        elif isinstance(X, np.ndarray):
            self.D = squareform(pdist(X,metric=self.metric))
        else:
            raise ValueError("you must provide either X or D as a numpy array")
        self.N = self.D.shape[0]
        if self.cutoff == 'auto':
            self.cutoff = self.use_percent()
        #calculate decision graph
        self.rho,self.order = self.calc_rho()
        self.delta,self.nneigh = self.calc_delta()
        return self.rho,self.delta

    def get_centroids(self,**kwargs):
        """
        search for points above level of mean 
        values of rho and delta for candidate 
        cluster centroids and for outliers
        """
        rmin = 0.0
        dmin = 0.0
        for key,value in kwargs.items():
            if key=="rmin":
                rmin = value
            if key=="dmin":
                dmin = value
        rmask = self.rho   > rmin
        dmask = self.delta > dmin
        self.centroids = np.where(rmask & dmask)[0]
        self.nclusters = len(self.centroids)
        self.points = np.where((~rmask) | (~dmask))[0]
        return self.centroids, self.points

    def assign_points(self):
        """
        assign a point to the same cluster 
        of its nearest neighbor with highest 
        density
        """
        self.clusters = -np.ones(self.N,dtype=np.int64)
        #initialize cluster labels to centroids
        for i in self.centroids:
            self.clusters[i] = i
        for point in self.order:
            if self.clusters[point] == -1:
                self.clusters[point] = self.clusters[self.nneigh[point]]
        if np.any(self.clusters==-1):
            raise ValueError("Error: Unassigned points are present")
        return self.clusters

    def create_halo(self):
        """
        create halo of points for each cluster that
        may be assigned as noise
        """
        self.robust_clusters = -np.ones(self.N,dtype=np.int64)
        rho_b = dict()
        for myc in self.centroids:
            rho_b[myc] = 0.0
            NN = np.asarray(list(range(self.N)))
            intpoints  = self.clusters == myc
            extpoints = ~intpoints
            intpoints = NN[intpoints]
            extpoints = NN[extpoints]
            for i in intpoints:
                for j in extpoints:
                    if self.D[i,j] <= self.cutoff:
                        rho_b[myc] = max(0.5*(self.rho[i]+self.rho[j]),rho_b[myc])
        for p,P in enumerate(self.clusters):
            if self.rho[p] >= rho_b[P]:
                self.robust_clusters[p] = self.clusters[p]
        return self.robust_clusters
