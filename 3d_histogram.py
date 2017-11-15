#Libraries
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import argparse
import time


#Variables for plotting
'''parser = argparse.ArgumentParser()
parser.add_argument("min", type=int, help="the lower bound for plotting (same for x,y and z)")
parser.add_argument("max", type=int, help="the upper bound for plotting (same for x,y and z)")
parser.add_argument("bins", type=int, help="the number of bins (same for x,y and z)")
parser.add_argument("data", type=int, help="the number of data points")
args = parser.parse_args()'''

'''nmin = args.min
nmax = args.max
nbins = args.bins
ndata = args.data
bin_size = (nmax-nmin)/float(nbins)'''

nmin = 0
nmax = 5
nbins = 100
ndata = 10
bin_size = (nmax-nmin)/float(nbins)


#Setting the data
data = np.random.multivariate_normal([2,4,3],np.identity(3)*0.1,ndata)


#3d gaussian kernel with mean: mu, factor(in exponential): inv2sigma2, calculated on the entire grid
def gd3DKernel(grid,mu,inv2sigma2): #63.18s
    r = np.zeros(len(grid))
    for point in range(0,len(grid)):
        prob = np.exp(-np.linalg.norm(grid[point]-mu)**2*inv2sigma2)
        if prob > 0.0005:
            r[point] += prob

    return r

def gd3DKernelFaster(grid,mu,sigma,inv2sigma2): #64.49s
    r = np.zeros(len(grid))
    for point in range(0,len(grid)):
        norm = np.linalg.norm(grid[point]-mu)
        if norm < 4*sigma:
            prob = np.exp(-norm**2*inv2sigma2)
            r[point] += prob

    return r

def gd3DKernelEvenFaster(grid,norms_grid,mu,norm_mu,sigma,inv2sigma2): #41.53s
    r = np.zeros(len(grid))
    for point in range(0,len(grid)):
        norm = np.sqrt(norms_grid[point]**2+norm_mu**2-2*np.dot(grid[point],mu))
        if norm < 4*sigma:
            prob = np.exp(-norm**2*inv2sigma2)
            r[point] += prob

    return r


#Making the histogram
def make_histogram(grid,data):
    hist = np.zeros(len(grid))
    sigma = 1
    inv2sigma2 = 1.0/(2.0*sigma**2)
    norms_grid = [np.linalg.norm(i) for i in grid]
    for d in data:
        norm_d = np.linalg.norm(d)
        hist += gd3DKernelEvenFaster(grid,norms_grid,d,norm_d,sigma,inv2sigma2)

    hist = hist/(np.sqrt(2.0*np.pi*sigma**2)*float(ndata))
    return hist


#Contour function to find points of array: hist, with only the probabilty: c
def contour(c,hist,delta=0.1):
    return [np.logical_and(hist < c+delta/2.0, hist > c-delta/2.0)]


#Plotting the results
fig = plt.figure()
ax = Axes3D(fig)

x = np.linspace(nmin,nmax,nbins)
grid = np.vstack(np.meshgrid(x,x,x)).reshape(3,-1).T

start = time.clock()
hist = make_histogram(grid,data)
print "#Time taken : ", time.clock() - start, "s"

grid = grid[contour(0.1,hist)]
print len(grid)
ax.scatter(grid[:,0],grid[:,1],grid[:,2])
plt.show()
