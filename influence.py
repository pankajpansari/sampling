import networkx as nx
import numpy as np
import sys
import time
import random
import torch
import os
import math
from torch.autograd import Variable

random.seed(1234)

class Influence(object):
    def __init__(self, G, p, niter):
        self.cache = {}
        self.cache_hits = 0
        self.G = G
        self.niter = niter 
        self.p = p 

    def reset(self):
        self.cache.reset()
        self.cache_hits = 0

    def __call__(self, sample):

#        one_hot = sample.byte() 
        key = sample.tobytes()

#        val = ic_model(self.G, sample)
#        self.cache[key] = val
        if key not in self.cache:
            val = ic_model(self.G, sample, self.p, self.niter)
            self.cache[key] = val
        else:
            self.cache_hits += 1

        return self.cache[key]

###########################################
def ic_model(G, sample, p, iterations):
#Independent Cascade Model

    N = G.number_of_nodes()

    seed_set = [i for i in range(N) if sample[i] == True]
    avg_influence = Variable(torch.FloatTensor([0]), requires_grad = True) 

    for j in range(iterations):            
        S=list(seed_set)
        for i in range(len(S)):                 # Process each node in seed set
            for neighbor in G.neighbors(S[i]):    
                if random.random()<p:           # Generate a random number and compare it with propagation probability
                    if neighbor not in S:       
                        S.append(neighbor)
        avg_influence = avg_influence + (float(len(S))/iterations) 
#    print 'Total influence:',int(round(avg_influence.item()))
    return avg_influence

def read_graph(filename):
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
    return G 

def main():
    #pick 10 graphs at random

    #pick 10 samples at random

    p = float(sys.argv[1])

    f = open('log_p_' + str(p) + '.txt', 'w')

    N_list = [6, 7, 8, 9, 10]
    niter_list = [10, 100, 1000]
    data_dir = '/home/pankaj/Sampling/data/input/social_graphs/k_'
    ngraphs = 4 
    nsamples = 5 

    for N in N_list:
        for iter_num in niter_list:
            graph_dir = data_dir + str(N) + '/'
            file_list = os.listdir(graph_dir)
            for i in range(ngraphs):
                G = read_graph(graph_dir + file_list[i])
                for j in range(nsamples):
                    sample = torch.rand(int(math.pow(2, N))) > 0.95
                    for k in range(20):
                        tic = time.clock()
                        val = ic_model(G, sample, p, iter_num).item()
                        toc = time.clock()
                        to_write_list = [N, iter_num, i, j, k, val, toc - tic]
                        print ' '.join(map(str, to_write_list)) + '\n'
                        f.write(' '.join(map(str, to_write_list)) + '\n')
if __name__ == '__main__':
    main()
