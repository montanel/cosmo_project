#Libraries
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import argparse
import time
from mpi4py import MPI



#3d gaussian kernel with mean: mu, factor(in exponential): inv2sigma2, calculated on the entire grid (benchmarks are with nbins=100,ndata=10)
def gd3DKernelPrevious(grid,norms_grid,mu,norm_mu,sigma2,inv2sigma2): #3.93 s
    hist = np.zeros(len(grid))
    norm = np.sqrt(norms_grid**2+norm_mu**2-2*np.dot(grid,mu))
    hist[np.less_equal(norm,4*sigma2)] += np.exp(-norm[np.less_equal(norm,4*sigma2)]**2*inv2sigma2)
    return hist

def gd3DKernel(grid,mu,sigma,inv2sigma2): #0.15 s
    global xvect
    hist = np.zeros(len(grid))

    x4sigma = xvect[np.less_equal(abs(xvect-mu[0]),4*sigma)]
    y4sigma = xvect[np.less_equal(abs(xvect-mu[1]),4*sigma)]
    z4sigma = xvect[np.less_equal(abs(xvect-mu[2]),4*sigma)]
    xexp = np.exp(-(x4sigma-mu[0])**2*inv2sigma2)
    yexp = np.exp(-(y4sigma-mu[1])**2*inv2sigma2)
    zexp = np.exp(-(z4sigma-mu[2])**2*inv2sigma2)

    vect = np.array([yexp, xexp, zexp])
    outerprod = reduce(np.multiply.outer,vect).flatten()
    hist[inBox(grid,4*sigma,mu)] = outerprod

    return hist

def gd3DKernelIselect(grid,mu,sigma,inv2sigma2):
    global xvect
    global nmin
    dx = abs(xvect[1]-xvect[0])
    hist = np.zeros(len(grid))

    x4sigma = xvect[int(np.rint((mu[0]-4*sigma-nmin)/dx)):int(np.rint((mu[0]+4*sigma-nmin)/dx))]
    y4sigma = xvect[int(np.rint((mu[1]-4*sigma-nmin)/dx)):int(np.rint((mu[1]+4*sigma-nmin)/dx))]
    z4sigma = xvect[int(np.rint((mu[2]-4*sigma-nmin)/dx)):int(np.rint((mu[2]+4*sigma-nmin)/dx))]
    xexp = np.exp(-(x4sigma-mu[0])**2*inv2sigma2)
    yexp = np.exp(-(y4sigma-mu[1])**2*inv2sigma2)
    zexp = np.exp(-(z4sigma-mu[2])**2*inv2sigma2)

    vect = np.array([yexp, xexp, zexp])
    outerprod = reduce(np.multiply.outer,vect).flatten()
    hist[inBox(grid,4*sigma,mu)] = outerprod

    return hist

#If we say that sigma2x=sigma2y=sigma2z=sigma2 in the covariance matrix and all
#the other elements are 0, we can simplify det(covariance)=sigma2**3
def make_histogram(grid,data):
    hist = np.zeros(len(grid))
    sigma = 1
    inv2sigma2 = 1.0/(2.0*sigma**2)
    for d in local_data:
        hist += gd3DKernel(grid,d,sigma,inv2sigma2)

    hist = hist/(np.sqrt((2.0*np.pi*sigma**2)**3)*float(ndata))
    return hist

#Contour function to find points of array: hist, with only the probabilty: c
def contour(c,hist,delta):
    return [np.logical_and(hist < c+delta, hist > c-delta)]

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

xvect = np.linspace(nmin,nmax,nbins)
grid = np.vstack(np.meshgrid(xvect,xvect,xvect)).reshape(3,-1).T
hist = np.zeros(len(grid))



#MPI variables
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#Setting the data
covmatrix = np.identity(3)*0.1
data = np.random.multivariate_normal([0,0,0],covmatrix,ndata) if rank == 0 else None

#Sharing
local_data = np.zeros((ndata/size,3))
comm.Scatter(data,local_data,root=0)



if rank == 0: start = time.clock()
local_hist = make_histogram(grid,local_data)
if rank == 0:
    time_taken = time.clock()-start
    print "#### Time taken for", size, "processe(s):", time_taken, "s ####"
    '''txtfile = open("/home/luca/Documents/COSMO/benchmarks_3d.txt","a")
    txtfile.write("%i %i %i %f\n" % (nbins**3,ndata,size,time_taken))
    txtfile.close()'''

comm.Reduce(local_hist,hist,op=MPI.SUM,root=0)



#Plotting the results
if rank == 0:
    fig = plt.figure()
    ax = Axes3D(fig)

    grid = grid[contour(c,hist,0.001)]
    ax.scatter(grid[:,0],grid[:,1],grid[:,2])
    ax.set_xlim(nmin,nmax)
    ax.set_xlabel("x axis")
    ax.set_ylim(nmin,nmax)
    ax.set_ylabel("y axis")
    ax.set_zlim(nmin,nmax)
    ax.set_zlabel("z axis")
    plt.show()
