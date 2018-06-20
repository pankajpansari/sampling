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
from influence import ic_model
from frank_wolfe import runFrankWolfe

random.seed(1234)

dirw = "/home/pankaj/Sampling/data/working/19_06_2018/"

def select_random_k_pair(G, k, nNodes, nRandom, p, num_influ_iter):

    val_list = []
    for t in range(nRandom):

        nodes = range(nNodes)
        sample = torch.zeros(nNodes)
        k_pair = np.random.choice(nodes, size = k, replace = False)
        sample[k_pair] = 1

        temp = ic_model(G, sample, p, num_influ_iter).item()
        val_list.append(temp)
#    print "Random function values = ", val_list
#    print "Random max = ", np.max(val_list)
    return np.max(val_list)

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
    return ic_model(G, top_k, p, num_influ_iter)
    #Compare with 10 randomly drawn k-pairs

def study_k_effect(filename, k, p, num_influ_iter, nsample_mlr, num_fw_iter, results, i):

    f = open(filename, 'rU')

    G = nx.DiGraph()

    for line in f:
        if line.find('Nodes') != -1:
            N = int(line.split(' ')[2])
            G.add_nodes_from(range(N))
            break

    for _ in range(1):
        next(f)

    for line in f:
        from_id = int(line.split()[0])
        to_id = int(line.split()[1])
        G.add_edge(from_id, to_id)

    N = 32
    random_val = select_random_k_pair(G, k, N, 5, p, num_influ_iter) 
    gt_val = get_ground_truth(G, k, nsample_mlr, num_fw_iter, p, num_influ_iter).item()
    results.append((i, k, gt_val, random_val))

def main():

    graph_dir = "/home/pankaj/Sampling/data/input/social_graphs/N_32/"
    file_list = os.listdir(graph_dir)
    graph_file_list = []

    for i in range(30):
        if 'log' not in file_list[i] and 'gt' not in file_list[i]:
            graph_file_list.append(file_list[i])
    
    N = 32
    p = float(sys.argv[1]) 
    num_influ_iter = 100
    nsample_mlr = 100
    num_fw_iter = 100


    params = [N, p, num_influ_iter, nsample_mlr, num_fw_iter]
    param_string = '_'.join(str(t) for t in params) 
    bufsize = 0
    f = open(dirw + 'log_k_study_' + param_string + '.txt', 'w', bufsize)

    for i in range(len(graph_file_list)):
#    for i in range(2):
        
        filename = graph_dir + graph_file_list[i] 
        results = []

        for k in range(1, 26):
            print i, k
            study_k_effect(filename, k, p, num_influ_iter, nsample_mlr, num_fw_iter, results, i)

        f.write('\n'.join('%s %s %s %s' % x for x in results))
        f.write('\n')
    f.close()

if __name__ == '__main__':
    main()


