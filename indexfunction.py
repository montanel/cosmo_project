#Libraries
import numpy as np
import time

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

nmin = -1
nmax = 1
nbins = 100

x = np.linspace(nmin,nmax,nbins)
grid = np.vstack(np.meshgrid(x,x,x)).reshape(3,-1).T
norm = np.array(np.linalg.norm(grid,axis=1))
print grid[index([-0.5,0,0.66],grid)]
