#Libraries
import random
import matplotlib.pyplot as plt
import numpy as np
import argparse
import time
import sys



#Variables for plotting
parser = argparse.ArgumentParser()
parser.add_argument("min", type=int, help="the lower x bound for plotting")
parser.add_argument("max", type=int, help="the upper x bound for plotting")
parser.add_argument("bins", type=int, help="the number of bins")
args = parser.parse_args()

start = time.clock()

xmin = args.min
xmax = args.max
nbins = args.bins
bin_size = (xmax-xmin)/float(nbins)


#Setting the data
data_rand = [random.gauss(6,1) for i in range(0,50000)]
nb_data = len(data_rand)


#Functions for the histogram
#rect evaluates the rectangular function for every point of grid
def rect(grid,data_val,bin_size):
    y = []
    step = data_val//bin_size*bin_size #step is the value at which the rectangle fct will raise
    middle_step = step + bin_size/2. #middle_step is the middle of the rectangle
    for i in grid:
        if (abs(i-middle_step) < bin_size/2. or i == step):
            y.append(1./bin_size)
        else: y.append(0)

    return y

#tri returns the value of a triangle function around data_val for every point of grid
def tri(grid,data_val):
    y = []
    for i in grid:
        if (i>data_val-1 and i<=data_val): #first half of the triangle (should 1 be replaced by bin_size ?)
            y.append(1-(data_val-i))
        elif (i>data_val and i<=data_val+1):
            y.append(1-(i-data_val))
        else: y.append(0)

    return y

def gdistr(grid,data_val,sigma, inv2sigma2):
    """
    Computes a gaussian with mean: data_val and width: sigma on the entire grid: grid
    """

    dx = np.abs(grid[1] - grid[0])
    mu = data_val - grid[0]
    r = grid * 0.0
    r[int((mu - 4 * sigma) / dx + 1): int((mu + 4 * sigma) / dx + 1)] += np.exp(-(grid[int((mu - 4 * sigma) / dx + 1): int((mu + 4 * sigma) / dx + 1)] - mu)**2)
     
    return r


#Making the histogram
def make_histogram(grid,data,choice='r',bin_size=None):
    hist = [0]*len(grid)
    if choice == 't':
        for i in data:
            hist += np.array(tri(grid,i))/float(nb_data)
    elif choice == 'g':
        sigma = 0.1
        inv2sigma2 =  1.0 / (2.0 * sigma**2)
        norm = 1.0 / np.sqrt(2.0 * np.pi * sigma**2)
        for i in data:
            hist += gdistr(grid,i,0.1, inv2sigma2)/float(nb_data)
        hist * norm
    elif choice == 'r':
        for i in data:
            hist += np.array(rect(grid,i,bin_size))/float(nb_data)
    return hist


#Plotting the results
grid = np.arange(xmin,xmax+1,bin_size)
#y1 = make_histogram(grid,data_rand,'r',0.2) #(the graph with this histogram goes beyond 1 in y values)
#y2 = make_histogram(grid,data_rand,'t')
y3 = make_histogram(grid,data_rand,'g')

print "# Time taken : ", time.clock() - start, "(s)"

plt.plot(grid,y3,'b')
plt.axis([xmin,xmax,0,1.2])
plt.show()


