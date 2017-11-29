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


def ref_f(grid):
    mu = 6.0
    sigma = 1.0
    return np.sqrt(2.0 * np.pi * sigma**2)**-1  * np.exp( -1.0 * (grid - mu)**2 / 2.0 / sigma**2)

#Making the histogram
def make_histogram(grid,data):
    hist = np.zeros(len(grid))
    sigma = 0.01
    inv2sigma2 =  1.0 / (2.0 * sigma**2)
    norm = 1.0 / np.sqrt(2.0 * np.pi * sigma**2)
    for i in data:
        hist += gdistr(grid, i, sigma, inv2sigma2)

    hist*=norm/float(nb_data)
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
args = parser.parse_args()

xmin = args.min
xmax = args.max
nbins = args.bins

hist = np.zeros(nbins)
grid = np.linspace(xmin,xmax,nbins)


#Setting the data
data_rand = np.array([random.gauss(6,1) for i in range(0,500000)]) if rank == 0 else None
nb_data = len(data_rand) if rank == 0 else None

#Sharing
nb_data = comm.bcast(nb_data,root=0)
local_data_rand = np.zeros(nb_data/size)
comm.Scatter(data_rand,local_data_rand,root=0)


#Calculating histogram in parallel
if rank == 0:
    start = time.clock()

local_hist = make_histogram(grid,local_data_rand)
comm.Reduce(local_hist,hist,op=MPI.SUM)

if rank == 0:
    print "#Time taken in parallel:", time.clock() - start, "s"


#Plotting the results
if rank == 0:
    plt.plot(grid, hist)

    plt.plot(grid, ref_f(grid))
    plt.axis([xmin,xmax,0,1.2])
    plt.show()
