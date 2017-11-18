#Libraries
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import argparse
import time


#Initialization
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

nmin = -2
nmax = 2
nbins = 100
ndata = 1000
bin_size = (nmax-nmin)/float(nbins)

x = np.linspace(nmin,nmax,nbins)
grid = np.vstack(np.meshgrid(x,x,x)).reshape(3,-1).T


#Setting the data
data = np.random.multivariate_normal([0,0,0],np.identity(3)*0.1,ndata)


#3d gaussian kernel with mean: mu, factor(in exponential): inv2sigma2, calculated on the entire grid (benchmarks are with nbins=100,ndata=10)
def gd3DKernel(grid,mu,inv2sigma2): #63.18s
    hist = np.zeros(len(grid))
    for point in range(0,len(grid)):
        prob = np.exp(-np.linalg.norm(grid[point]-mu)**2*inv2sigma2)
        if prob > 0.0005:
            hist[point] += prob

    return hist

def gd3DKernelFaster(grid,mu,sigma2,inv2sigma2): #64.49s
    hist = np.zeros(len(grid))
    for point in range(0,len(grid)):
        norm = np.linalg.norm(grid[point]-mu)
        if norm < 4*sigma2:
            hist[point] += np.exp(-norm**2*inv2sigma2)

    return hist

def gd3DKernelEvenFaster(grid,norms_grid,mu,norm_mu,sigma2,inv2sigma2): #41.53s
    hist = np.zeros(len(grid))
    for point in range(0,len(grid)):
        norm = np.sqrt(norms_grid[point]**2+norm_mu**2-2*np.dot(grid[point],mu))
        if norm < 4*sigma2:
            hist[point] += np.exp(-norm**2*inv2sigma2)

    return hist

def gd3DKernelFastest(grid,norms_grid,mu,norm_mu,sigma2,inv2sigma2): #3.93 s
    hist = np.zeros(len(grid))
    norm = np.sqrt(norms_grid**2+norm_mu**2-2*np.dot(grid,mu))
    hist[np.less_equal(norm,4*sigma2)] += np.exp(-norm[np.less_equal(norm,4*sigma2)]**2*inv2sigma2)
    return hist


#Making the histogram
#If we say that sigma2x=sigma2y=sigma2z=sigma2 in the covariance matrix and all
#the other elements are 0, we can simplify det(covariance)=sigma2**3
def make_histogram(grid,data):
    hist = np.zeros(len(grid))
    sigma2 = 1
    inv2sigma2 = 1.0/(2.0*sigma2)
    norms_grid = np.array([np.linalg.norm(i) for i in grid])
    for d in data:
        norm_d = np.linalg.norm(d)
        hist += gd3DKernelFastest(grid,norms_grid,d,norm_d,sigma2,inv2sigma2)

    hist = hist/(np.sqrt((2.0*np.pi*sigma2)**3)*float(ndata))
    return hist


#Contour function to find points of array: hist, with only the probabilty: c
def contour(c,hist,delta):
    return [np.logical_and(hist < c+delta/2.0, hist > c-delta/2.0)]


#Function to verify if histogram is well normalized
def isosphere(c,delta):
    ceff = c+delta/2.0
    r = 2*0.1*np.log(ceff*np.sqrt((2*np.pi*0.1)**3))
    phi, theta = np.mgrid[0.0:0.5*np.pi:180j, 0.0:2.0*np.pi:720j]
    X = r*np.sin(phi)*np.cos(theta)
    Y = r*np.sin(phi)*np.sin(theta)
    Z = r*np.cos(phi)
    ax.plot_surface(X,Y,Z)


start = time.clock()
hist = make_histogram(grid,data)
print "#Time taken:", time.clock() - start, "s"

#Plotting the results
fig = plt.figure()
ax = Axes3D(fig)

grid = grid[contour(0.1,hist,0.1)]
ax.scatter(grid[:,0],grid[:,1],grid[:,2])
ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
ax.set_zlim(-1,1)
plt.show()
