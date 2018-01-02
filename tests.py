#Libraries
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def index(pt,grid):
    dx = abs(grid[1,2]-grid[0,2])
    nmin = min(grid[:,2])
    nbins = np.cbrt(len(grid))
    return int(np.around((pt[2]-nmin)/dx)+np.around((pt[0]-nmin)/dx*nbins,decimals=-2)+np.around((pt[1]-nmin)/dx*nbins**2,decimals=-4))

def indexboolversion(pt,grid):
    dx = abs(grid[1,2]-grid[0,2])
    nmin = min(grid[:,2])
    nbins = np.cbrt(len(grid))
    index = int(np.around((pt[2]-nmin)/dx)+np.around((pt[0]-nmin)/dx*nbins,decimals=-2)+np.around((pt[1]-nmin)/dx*nbins**2,decimals=-4))
    boolgrid = np.zeros(len(grid),dtype=bool)
    boolgrid[index] = True
    return boolgrid

def inBox(grid,side,datapt):
    return np.logical_and(np.logical_and(np.less_equal(abs(grid[:,0]-datapt[0]),side),np.less_equal(abs(grid[:,1]-datapt[1]),side)),np.less_equal(abs(grid[:,2]-datapt[2]),side))

nmin = -1
nmax = 1
nbins = 50

x = np.linspace(nmin,nmax,nbins)
grid = np.vstack(np.meshgrid(x,x,x)).reshape(3,-1).T
datapt = [0,0,0]
sigma = 0.1

fig = plt.figure()
ax = Axes3D(fig)

grid = grid[inBox(grid,4*sigma,datapt)]
ax.scatter(grid[:,0],grid[:,1],grid[:,2])
ax.set_xlim(nmin,nmax)
ax.set_ylim(nmin,nmax)
ax.set_zlim(nmin,nmax)
plt.show()
