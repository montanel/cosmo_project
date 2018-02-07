#Libraries
import random
import matplotlib.pyplot as plt
import numpy as np
import argparse
import time
from mpi4py import MPI

#Functions for the histogram
#Computes a gaussian with mean: data_val and width: sigma on the entire grid: grid
def gdistr(grid,data_val,sigma, inv2sigma2):
    dx = np.abs(grid[1] - grid[0])
    mu = data_val - grid[0]
    r = grid * 0.0
    r[int((mu - 4 * sigma) / dx + 1): int((mu + 4 * sigma) / dx + 1)] += np.exp(-(grid[int((mu - 4 * sigma) / dx + 1): int((mu + 4 * sigma) / dx + 1)] - mu)**2 * inv2sigma2)

    return r

#Making the histogram
def make_histogram(grid,data,sigma):
    hist = np.zeros(len(grid))
    inv2sigma2 =  1.0 / (2.0 * sigma**2)
    norm = 1.0 / np.sqrt(2.0 * np.pi * sigma**2)
    for i in data:
        hist += gdistr(grid, i, sigma, inv2sigma2)

    hist*=norm/float(ndata)
    return hist

#MPI variables
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#Variables for plotting
parser = argparse.ArgumentParser()
parser.add_argument("min", type=int, help="the lower x bound for plotting")
parser.add_argument("max", type=int, help="the upper x bound for plotting")
parser.add_argument("bins", type=int, help="the number of bins")
parser.add_argument("data", type=int, help="the number of data points")
parser.add_argument("mean", type=float, help="the mean of the normal distribution")
parser.add_argument("std", type=float, help="the standard deviation of the normal distribution")
parser.add_argument("sigma", type=float, help="the standard deviation of the gaussian kernel")
args = parser.parse_args()

xmin = args.min
xmax = args.max
nbins = args.bins
ndata = args.data
mean = args.mean
std = args.std
sigma = args.sigma

hist = np.zeros(nbins)
grid = np.linspace(xmin,xmax,nbins)


#Setting the data
data_rand = np.array([random.gauss(mean,std) for i in range(0,ndata)]) if rank == 0 else None

#Sharing
local_data_rand = np.zeros(ndata/size)
comm.Scatter(data_rand,local_data_rand,root=0)


#Calculating histogram in parallel
if rank == 0: start = time.clock()
local_hist = make_histogram(grid,local_data_rand,sigma)
if rank ==0:
    time_taken = time.clock()-start
    print "#### Time taken for", size, "processe(s):", time_taken, "s ####"
    '''txtfile = open("/home/luca/Documents/COSMO/benchmarks_1d.txt","a")
    txtfile.write("%i %i %i %f\n" % (nbins,ndata,size,time_taken))
    txtfile.close()'''

comm.Reduce(local_hist,hist,op=MPI.SUM)


#Plotting the results
if rank == 0:
    plt.plot(grid, hist)
    plt.axis([xmin,xmax,0,1.2])
    plt.show()
