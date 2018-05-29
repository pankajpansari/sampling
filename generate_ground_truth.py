import networkx as nx
import sys
import numpy as np
import math
import torch
from influence import ic_model as submodObj
from torch.autograd import Variable
from frank_wolfe import runFrankWolfe, getGrad
import time

#filename = '/home/pankaj/Sampling/data/input/social_graphs/train/g_k_7_0-network.txt'
filename = '/home/pankaj/Sampling/data/input/social_graphs/g_k_5_0-network.txt'
f = open(filename, 'rU')
nNodes = 0
for line in f:
    if line == '\n':
        break
    else:
        nNodes += 1

alphas = np.zeros((nNodes, nNodes))

for line in f:
    temp = line.strip('\n')
    temp2 = temp.split(",")
    a = int(temp2[0])
    b = int(temp2[1])
    alphas[a, b] = float(temp2[2])

max_alpha = np.max(alphas)
#alphas are affinity measures - take -ve to get a dissimilarity measure
weights = max_alpha - alphas
w = np.asmatrix(weights)
G = nx.DiGraph(w)

N = nx.number_of_nodes(G)

runFrankWolfe(G, 10)

#tic = time.clock()
#for t in range(100):
#    sample = Variable(torch.bernoulli(torch.rand(N)))
#    submodObj(G, sample)
#toc = time.clock()
#print "Time elapsed = ", toc - tic
#print "Average time per call = ", (toc - tic)/100
#
#tic = time.clock()
#x = torch.rand(N)
#grad = getGrad(G, x, 100)
#toc = time.clock()
#print "Time elapsed = ", toc - tic
#sys.exit()
#count = 0
#for t in sample.data:
#    if t == 1:
#        count += 1
#
#submodObj(G, sample)


