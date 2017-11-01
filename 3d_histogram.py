#Libraries
import random
import matplotlib.pyplot as plt
import numpy as np
import argparse
import time


#Variables for plotting
parser = argparse.ArgumentParser()
parser.add_argument("min", type=int, help="the lower x bound for plotting")
parser.add_argument("max", type=int, help="the upper x bound for plotting")
parser.add_argument("bins", type=int, help="the number of bins")
parser.add_argument("data", type=int, help="the number of data points")
args = parser.parse_args()

xmin = args.min
xmax = args.max
nbins = args.bins
ndata = args.data
bin_size = (xmax-xmin)/float(nbins)


#Setting the data
data = [random.gauss(6,1) for i in range(0,ndata)]


#3d histogram fct for histogram
#Computes a gaussian with mean: data_val and width: sigma on the entire grid: grid
def gdistr(grid,data_val,sigma, inv2sigma2):
    dx = np.abs(grid[1] - grid[0])
    mu = data_val - grid[0]
    r = grid * 0.0
    r[int((mu - 4 * sigma) / dx + 1): int((mu + 4 * sigma) / dx + 1)] += np.exp(-(grid[int((mu - 4 * sigma) / dx + 1): int((mu + 4 * sigma) / dx + 1)] - mu)**2)

    return r


#Making the histogram
def make_histogram(grid,data):
    hist = [0]*len(grid)
    sigma = 0.1
    inv2sigma2 =  1.0 / (2.0 * sigma**2)
    norm = 1.0 / np.sqrt(2.0 * np.pi * sigma**2)
    for i in data:
            hist += gdistr(grid,i,0.1,inv2sigma2)/float(nb_data)

    hist * norm
    return hist


#Plotting the results
grid = np.arange(xmin,xmax+1,bin_size)
start = time.clock()
y = make_histogram(grid,data)
print "#Time taken : ", time.clock() - start, "(s)"

plt.plot(grid,y,'b')
plt.axis([xmin,xmax,0,1.2])
plt.show()
