#!/usr/bin/env python

from __future__ import division
# import modules used here -- sys is a very standard one
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats 

def plotHistogram(a, start, end):

    x = a.flatten()
    print x.shape
    minVal = x.min()
    maxVal = x.max()
    print minVal
    print maxVal
    modeVal = stats.mode(x)[0]
    print modeVal 
    t = (end*minVal - start*maxVal)/(end - start)
    p = (maxVal - minVal)/(end - start)
    y = (x - t)/p
    transformy = np.tan((y + 0.5)*math.pi)
    transformy = np.log(y+start)
    plt.hist(transformy, 1000, log=True)

    plt.title("Negative Gradient Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
#    plt.savefig('gradient_histogram_transform.pdf')
    plt.show()
    
def plotHeatmap(a):
    print a.shape
    category = 1 
#    for category in range(21):
    b = a[category, :]
    b = b.reshape(320, 213) 
    print b.shape
    plt.imshow(b, cmap = 'hot', interpolation = 'nearest')
    plt.colorbar()
    plt.savefig('negGrad_heatmap_grass.pdf')
#    plt.show()

def linePlot(y, z, filename):
     plt.plot(y, linestyle='-', color = 'r', label = 'training') 
     plt.plot(z, linestyle='-', color = 'b', label = 'validation') 
     plt.xlabel('Iterations')
     plt.ylabel('KL-divergence')
     plt.legend()
     plt.show()

def linePlotShow(x):
     plt.plot(x, linestyle='-') 
     plt.xlabel('Iterations')
     plt.ylabel('Obj function val')
#     plt.ylim([0, 100])
#     plt.axhline(y[0], linewidth=2, color = 'b')
#     plt.xlim([-1, 1])
     plt.show()

def difference(a):
     b = a[a[:, 0] == 0].shape[0]
     diff = np.zeros(b)
     ngraphs = 2
     for i in range(ngraphs):
         ind = a[:, 0] == i 
         diff += a[ind, 2] - a[ind, 3]
     diff = diff/ngraphs
     for i in range(len(diff)):
         print i+1, diff[i]
     plt.plot(diff)
     plt.xlabel('Cardinality Constraint')
     plt.ylabel('Difference')
     plt.title('Plot of difference in submodular function values between optimum and best of 20 randomly drawn sets (averaged over 8 graphs of 32 nodes)')
     plt.savefig(sys.argv[2])

# Gather our code in a main() function
def main():
     filename = sys.argv[1]
     f = open(filename, 'r')
     val = []
     for line in f:
	print line
 	a = line.split(' ')
	val.append(float(a[1]))
     linePlotShow(val)

if __name__ == '__main__':
    main()
