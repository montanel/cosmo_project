#Libraries
import random
import matplotlib.pyplot as plt
import numpy as np
import argparse
import time
#from mpi4py import rc
#rc.finalize = False #to ensure that mpi4py doesn't initialize right away
from mpi4py import MPI


#MPI variables
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#Variables for plotting
'''parser = argparse.ArgumentParser()
parser.add_argument("min", type=int, help="the lower x bound for plotting")
parser.add_argument("max", type=int, help="the upper x bound for plotting")
parser.add_argument("bins", type=int, help="the number of bins")
args = parser.parse_args()

xmin = args.min
xmax = args.max
nbins = args.bins
bin_size = (xmax-xmin)/float(nbins)'''

xmin = 0
xmax = 10
nbins = 100*8
bin_size = (xmax-xmin)/float(nbins)

grid = np.arange(xmin,xmax+1,bin_size)


#Setting the data
if comm.rank == 0:
    data_rand = [random.gauss(6,1) for i in range(0,50000)]
    nb_data = len(data_rand)
else:
    data_rand = np.array([0])


#Functions for the histogram
#Computes a gaussian with mean: data_val and width: sigma on the entire grid: grid
def gdistr(grid,data_val,sigma, inv2sigma2):
    dx = np.abs(grid[1] - grid[0])
    mu = data_val - grid[0]
    r = grid * 0.0
    r[int((mu - 4 * sigma) / dx + 1): int((mu + 4 * sigma) / dx + 1)] += np.exp(-(grid[int((mu - 4 * sigma) / dx + 1): int((mu + 4 * sigma) / dx + 1)] - mu)**2)

    return r


#Making the histogram
def make_histogram_parallel(grid,data):
    hist = [0]*len(grid)
    sigma = 0.1
    inv2sigma2 =  1.0 / (2.0 * sigma**2)
    norm = 1.0 / np.sqrt(2.0 * np.pi * sigma**2)
    for i in data:
        hist += gdistr(grid,i,0.1, inv2sigma2)

    hist*norm/float(nb_data)
    return hist


#Caulating histogram in parallel
start = time.clock()

comm.Scatter(grid,local_grid,root=0)

comm.Scatter(grid,grid)
y = make_histogram_parallel(local_grid,data_rand)
MPI.Finalize()

print "#Time taken:", time.clock() - start, "s"


#Plotting the results
plt.plot(grid,y)
plt.axis([xmin,xmax,0,1])
plt.show()
