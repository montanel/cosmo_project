#Libraries
import numpy as np
import time

def index(pt,grid):
    dx = abs(grid[1,2]-grid[0,2])
    nmin = min(grid[:,2])
    nbins = np.cbrt(len(grid))
    return int(np.around((pt[2]-nmin)/dx)+np.around((pt[0]-nmin)/dx*nbins,decimals=-2)+np.around((pt[1]-nmin)/dx*nbins**2,decimals=-4))

nmin = -1
nmax = 1
nbins = 100

x = np.linspace(nmin,nmax,nbins)
grid = np.vstack(np.meshgrid(x,x,x)).reshape(3,-1).T
norm = np.array(np.linalg.norm(grid,axis=1))
print grid[np.less_equal(norm,1)]
print index(grid[np.less_equal(norm,1)],grid)
