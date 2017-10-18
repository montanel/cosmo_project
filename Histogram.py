import random
import matplotlib.pyplot as plt
import numpy as np

#Setting the data
data_rand = [random.gauss(6,1) for i in range(0,1000)]

#Functions for the histogram
def rect(x_eval,step):
    y = []
    middle_step = step + 0.5
    for i in x_eval:
        if (abs(i-middle_step) < 0.5 or i == step):
            y.append(1)
        else: y.append(0)

    return y

def tri(x_eval,peak):
    y = []
    for i in x_eval:
        if (i>peak-1 and i<=peak):
            y.append(1-(peak-i))
        elif (i>peak and i<=peak+1):
            y.append(1-(i-peak))
        else: y.append(0)

    return y

def gdistr(x_eval,mu,sigma,test=None):
    y = []
    if test == 'test':
        for i in x_eval:
            y.append(1000*np.exp(-(i-mu)**2/(2*sigma**2))/np.sqrt(2*np.pi*sigma**2))
    else:
        for i in x_eval:
            y.append(np.exp(-(i-mu)**2/(2*sigma**2))/np.sqrt(2*np.pi*sigma**2))

    return y

#Making the histogram
def make_histogram(x_values,data,choice='r',step=1):
    hist = [0]*len(x)
    if choice == 't':
        for i in data:
            hist += np.array(tri(x_values,i))
    elif choice == 'g':
        for i in data:
            hist += np.array(gdistr(x_values,i,np.std(data)))
    elif choice == 'r':
        for i in data:
            hist += np.array(rect(x_values/step,i//step))

    return hist

#Plotting the results
x = np.arange(0.0,max(data_rand),0.02)
y1 = gdistr(x,6,1,'test')
y2 = make_histogram(x,data_rand,'r')
y3 = make_histogram(x,data_rand,'t')
y4 = make_histogram(x,data_rand,'g')

plt.plot(x,y1,'k',x,y2,'r',x,y3,'g',x,y4,'b')
plt.axis([0,max(x),0,1.2*max(y1)])
plt.show()
