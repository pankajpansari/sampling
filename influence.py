import networkx as nx
import numpy as np
import sys
import time
import random
import torch
import os
import math
import itertools as it
import matplotlib.pyplot as plt
from torch.autograd import Variable

random.seed(1234)

dirw = "/home/pankaj/Sampling/data/working/19_06_2018/"

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

def read_graph(filename, nNodes):

    f = open(filename, 'rU')
    print filename
    adj = np.zeros((nNodes, nNodes))

    for line in f:
        if line.find('Nodes') != -1:
            N = int(line.split(' ')[2])
            assert(N == nNodes)
            break

    for _ in range(1):
        next(f)

    for line in f:
        from_id = int(line.split()[0])
        to_id = int(line.split()[1])
        adj[from_id, to_id] = 1


    f.close()

    return nx.DiGraph(adj)

def enumerate(k):
    ##Cannot be done quickly enough for larger k
    nodes = range(32)
    all_k_pairs = list(it.combinations(nodes, k))
    print len(all_k_pairs)

    graph_dir = "/home/pankaj/Sampling/data/input/social_graphs/N_32/"

    file_list = os.listdir(graph_dir)
    graph_file_list = []
    for i in range(30):
        if 'log' not in file_list[i] and 'gt' not in file_list[i]:
            graph_file_list.append(file_list[i])

    for i in range(2):
        print i
        G = read_graph(graph_dir + graph_file_list[i], 32)
        val_list = []
        count = 0
        for pair in all_k_pairs:
            sample = torch.zeros(32)
            sample[pair[0]] = 1
            sample[pair[1]] = 1
            temp = ic_model(G, sample, 0.1, 100).item()
            count += 1
            val_list.append(temp)
        val_list.sort() 
        for t in range(1, 6):
            print val_list[len(val_list) - t],
        print val_list[0]
#        plt.boxplot(val_list)
#    plt.show()
#    plt.savefig('boxplot_32.jpg')

def variance_study():
    #pick 10 graphs at random

    #pick 10 samples at random

    p = float(sys.argv[1])

    bufsize = 0
    f = open('variance_study_p_' + str(p) + '.txt', 'w', bufsize)

    niter_list = [10, 100, 1000]
    ngraphs = 4 
    nsamples = 5 

    graph_dir = "/home/pankaj/Sampling/data/input/social_graphs/N_32/"

    file_list = os.listdir(graph_dir)
    graph_file_list = []

    for i in range(30):
        if 'log' not in file_list[i] and 'gt' not in file_list[i]:
            graph_file_list.append(file_list[i])

    for iter_num in niter_list:
        for i in range(ngraphs):
            G = read_graph(graph_dir + graph_file_list[i], 32)
            for j in range(nsamples):
                sample = torch.rand(32) > 0.7
                val = []
                for k in range(20):
                    val.append(ic_model(G, sample, p, iter_num).item())
                to_write_list = [iter_num, i, j, np.var(val)]
                print ' '.join(map(str, to_write_list)) + '\n'
                f.write(' '.join(map(str, to_write_list)) + '\n')

def get_ground_truth(G, k, nsamples_mlr, num_fw_iter, p, num_influ_iter):

    N = nx.number_of_nodes(G)

#    ind = filename.find('.')
#    file_prefix = filename[0:ind] + '_' + str(k) + '_' + str(nsamples_mlr) + '_' + str(num_fw_iter) + '_' + str(p) + '_' + str(num_influ_iter) 
    file_prefix = 'temp'
    x_opt = runFrankWolfe(G, nsamples_mlr, k, file_prefix, num_fw_iter, p, num_influ_iter)
    #Round the optimum solution and get function values
    top_k = Variable(torch.zeros(N)) #conditional grad
    sorted_ind = torch.sort(x_opt, descending = True)[1][0:k]
    top_k[sorted_ind] = 1
#    print x_opt, top_k
    return submodObj(G, top_k, p, num_influ_iter)
    #Compare with 10 randomly drawn k-pairs

def main():
    filename = "/home/pankaj/Sampling/data/input/social_graphs/N_32/g_N_32_410.txt"
    G = read_graph(filename, 32)
    nNodes = 32
    k = 25
    nodes = range(nNodes)
    sample = torch.zeros(nNodes)
    k_pair = np.random.choice(nodes, size = k, replace = False)
    print len(k_pair)
    print k_pair
    sample[k_pair] = 1
    print sample
    temp = ic_model(G, sample, 0.01, 1).item()
    print temp 

if __name__ == '__main__':
    variance_study()
