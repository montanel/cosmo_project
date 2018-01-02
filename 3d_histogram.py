#Libraries
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import argparse
import time
from mpi4py import MPI


#3d gaussian kernel with mean: mu, factor(in exponential): inv2sigma2, calculated on the entire grid (benchmarks are with nbins=100,ndata=10)
def gd3DKernel(grid,norms_grid,mu,norm_mu,sigma2,inv2sigma2): #3.93 s
    hist = np.zeros(len(grid))
    norm = np.sqrt(norms_grid**2+norm_mu**2-2*np.dot(grid,mu))
    hist[np.less_equal(norm,4*sigma2)] += np.exp(-norm[np.less_equal(norm,4*sigma2)]**2*inv2sigma2)
    return hist


#try if np.less_equal(abs(xaxis-mu[0]),4*sigma) or int((mu[0]-4*sigma-nmin)/dx+1):int((mu[0]+4*sigma-nmin)/dx+1) is faster
def gd3DKernelCube(grid,mu,sigma2,inv2sigma2):
    hist = np.zeros(len(grid))
    nbins = int(np.cbrt(len(grid)))
    nmin = min(grid[:,2])
    nmax = max(grid[:,2])
    xaxis = yaxis = zaxis = np.array(np.linspace(nmin,nmax,nbins))
    xexp = yexp = zexp = np.zeros(nbins)

    x4sigma = xexp[np.less_equal(abs(xaxis-mu[0]),4*sigma2)]
    y4sigma = yexp[np.less_equal(abs(yaxis-mu[1]),4*sigma2)]
    z4sigma = zexp[np.less_equal(abs(zaxis-mu[2]),4*sigma2)]
    x4sigma = np.exp(-(xaxis[np.less_equal(abs(xaxis-mu[0]),4*sigma2)]-mu[0])**2*inv2sigma2)
    y4sigma = np.exp(-(yaxis[np.less_equal(abs(yaxis-mu[1]),4*sigma2)]-mu[1])**2*inv2sigma2)
    z4sigma = np.exp(-(zaxis[np.less_equal(abs(zaxis-mu[2]),4*sigma2)]-mu[2])**2*inv2sigma2)

    vect = np.array([y4sigma, x4sigma, z4sigma])
    outerprod = reduce(np.multiply.outer,vect).flatten()
    hist[inBox(grid,4*sigma2,mu)] = outerprod

    return hist


#Making the histogram
#If we say that sigma2x=sigma2y=sigma2z=sigma2 in the covariance matrix and all
#the other elements are 0, we can simplify det(covariance)=sigma2**3
def make_histogram(grid,data):
    hist = np.zeros(len(grid))
    sigma2 = 1
    inv2sigma2 = 1.0/(2.0*sigma2)
    norms_grid = np.array([np.linalg.norm(i) for i in grid])
    for d in local_data:
        norm_d = np.linalg.norm(d)
        #hist += gd3DKernelCube(grid,d,sigma2,inv2sigma2)
        hist += gd3DKernel(grid,norms_grid,d,norm_d,sigma2,inv2sigma2)

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


#Returns True if the point of grid is inside the box of sides 2*side centered around datapt
def inBox(grid,side,datapt):
    return np.logical_and(np.logical_and(np.less_equal(abs(grid[:,0]-datapt[0]),side),np.less_equal(abs(grid[:,1]-datapt[1]),side)),np.less_equal(abs(grid[:,2]-datapt[2]),side))



#MPI variables
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


#Initialization
parser = argparse.ArgumentParser()
parser.add_argument("min", type=int, help="the lower bound for plotting (same for x,y and z)")
parser.add_argument("max", type=int, help="the upper bound for plotting (same for x,y and z)")
parser.add_argument("bins", type=int, help="the number of bins (same for x,y and z)")
parser.add_argument("data", type=int, help="the number of data points")
parser.add_argument("contour", type=float, help="the value for the contour plot")
args = parser.parse_args()
nmin = args.min
nmax = args.max
nbins = args.bins
ndata = args.data
c = args.contour


x = np.linspace(nmin,nmax,nbins)
grid = np.vstack(np.meshgrid(x,x,x)).reshape(3,-1).T
hist = np.zeros(len(grid))

#Setting the data
data = np.random.multivariate_normal([0,0,0],np.identity(3)*0.1,ndata) if rank == 0 else None

#Sharing
local_data = np.zeros((ndata/size,3))
comm.Scatter(data,local_data,root=0)

start = time.clock()
local_hist = make_histogram(grid,local_data)
print "#Time taken for making local_hist:", time.clock() - start, "s"

comm.Reduce(local_hist,hist,op=MPI.SUM,root=0)

#Plotting the results
if rank == 0:
    fig = plt.figure()
    ax = Axes3D(fig)

    grid = grid[contour(c,hist,0.1)]
    ax.scatter(grid[:,0],grid[:,1],grid[:,2])
    ax.set_xlim(nmin,nmax)
    ax.set_ylim(nmin,nmax)
    ax.set_zlim(nmin,nmax)
    plt.show()
