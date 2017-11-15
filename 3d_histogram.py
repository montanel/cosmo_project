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
nbins = 10
ndata = 10
bin_size = (nmax-nmin)/float(nbins)


#Setting the data
data = np.random.multivariate_normal([2,4,3],np.identity(3)*0.1,ndata)


#3d histogram fct for histogram
#Computes a gaussian with mean: data_val and width: sigma on the entire grid: grid
def gd3DKernel(grid,mu,inv2sigma2):
    nbpoints = len(grid)
    r = [0]*len(grid)
    for point in range(0, nbpoints):
        prob = np.exp(-np.linalg.norm(grid[point]-mu)**2*inv2sigma2)
        if prob>0.0005:
            r[point] += prob

    return r


#Making the histogram
def make_histogram(grid,data):
    hist = [0]*len(grid)
    sigma = 1
    inv2sigma2 =  1.0 / (2.0 * sigma**2)
    norm = 1.0 / (np.sqrt(2.0 * np.pi * sigma**2)*float(ndata))
    for i in data:
        hist += gd3DKernel(grid,i,inv2sigma2)

    hist = [norm*x for x in hist]
    return hist


#Contour function
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

print hist
print contour(0.01,hist)
grid[contour(0.01,hist)]
print len(grid)
ax.scatter(grid[:,0],grid[:,1],grid[:,2])
plt.show()
