import networkx as nx
import sys
import numpy as np
import math
import torch
from influence import ic_model as submodObj
from torch.autograd import Variable
from frank_wolfe import runFrankWolfe, getGrad
import time

np.random.seed(1234)
torch.manual_seed(1234) 
#filename = '/home/pankaj/Sampling/data/input/social_graphs/train/g_k_7_0-network.txt'

filename = '/home/pankaj/Sampling/data/input/social_graphs/k_5/g_k_5_0-network.txt'

k = 5 #cardinality constraint
nsamples_mlr = 20 #draw these many sets from x for multilinear relaxation
num_fw_iter = 20
p = 0.01
num_influ_iter = 100

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

ind = filename.find('-')
file_prefix = filename[0:ind]

runFrankWolfe(G, nsamples_mlr, k, file_prefix, num_fw_iter, p, num_influ_iter)

print "Ground truth written to: " + file_prefix + "_gt.txt"
print "Log written to: " + file_prefix + "_log.txt"

