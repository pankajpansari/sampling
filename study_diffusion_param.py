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
from influence import Influence 
from frank_wolfe import runFrankWolfe
from frank_wolfe import getRelax

random.seed(1234)

dirw = "/home/pankaj/Sampling/data/working/23_07_2018/"

def select_random_k_pair(G, k, nNodes, nRandom, p, num_influ_iter):

    val_list = []
    for t in range(nRandom):

        nodes = range(nNodes)
        sample = torch.zeros(nNodes)
        k_pair = np.random.choice(nodes, size = k, replace = False)
        sample[k_pair] = 1

        temp = ic_model(G, sample, p, num_influ_iter).item()
        val_list.append(temp)
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

    p = float(sys.argv[1])
    nNodes = 16384 
    bufsize = 0
    f = open('variance_study_p_' + str(p) + '_N_' + str(nNodes) + '.txt', 'w', bufsize)

    niter_list = [10, 100, 1000]
    ngraphs = 4 
    nsamples = 10 

    graph_dir = "/home/pankaj/Sampling/data/input/social_graphs/N_" + str(nNodes) + "/"

    file_list = os.listdir(graph_dir)
    graph_file_list = []

    for i in range(ngraphs):
        if 'log' not in file_list[i] and 'gt' not in file_list[i]:
            graph_file_list.append(file_list[i])

    for i in range(ngraphs):
        for iter_num in niter_list:
            G = read_graph(graph_dir + graph_file_list[i], nNodes)
            for j in range(nsamples):
                sample = torch.rand(nNodes) > 0.8
                val = []
                tic = time.clock()
                for k in range(20):
                    val.append(ic_model(G, sample, p, iter_num).item())
                to_write_list = [iter_num, i, j, np.var(val), (time.clock() - tic)/20]
                print ' '.join(map(str, to_write_list)) + '\n'
                f.write(' '.join(map(str, to_write_list)) + '\n')

def influence_variance_study_parallel():

    p = float(sys.argv[1])
    file_id = int(sys.argv[2])
    nNodes = 512 
    bufsize = 0
    f = open(dirw + 'variance_study_p_' + str(p) + '_N_' + str(nNodes) + '_' + str(file_id) + '.txt', 'w', bufsize)

    niter_list = [10, 100, 1000, 10000]
    ngraphs = 10 

    graph_dir = "/home/pankaj/Sampling/data/input/social_graphs/N_" + str(nNodes) + "/"

    file_list = os.listdir(graph_dir)
    graph_file_list = []

    for i in range(ngraphs):
        if 'log' not in file_list[i] and 'gt' not in file_list[i]:
            graph_file_list.append(file_list[i])

    for iter_num in niter_list:
        G = read_graph(graph_dir + graph_file_list[file_id], nNodes)
        sample = torch.rand(nNodes) > 0.8
        val = []
        tic = time.clock()
        for k in range(20):
            val.append(ic_model(G, sample, p, iter_num).item())
        to_write_list = [iter_num, '0.8', file_id, np.var(val), (time.clock() - tic)/20]
        print ' '.join(map(str, to_write_list)) + '\n'
        f.write(' '.join(map(str, to_write_list)) + '\n')

    for iter_num in niter_list:
        G = read_graph(graph_dir + graph_file_list[file_id], nNodes)
        sample = torch.rand(nNodes) > 0.5
        val = []
        tic = time.clock()
        for k in range(20):
            val.append(ic_model(G, sample, p, iter_num).item())
        to_write_list = [iter_num, '0.5', file_id, np.var(val), (time.clock() - tic)/20]
        print ' '.join(map(str, to_write_list)) + '\n'
        f.write(' '.join(map(str, to_write_list)) + '\n')

def multilinear_variance_study():

    file_id = int(sys.argv[1])
    nsamples = int(sys.argv[2])
    nNodes = 512 
    bufsize = 0
    p = 0.4 
    num_influ_iter = 100

    f = open(dirw + 'multilinear_variance_study_p_' + str(p) + '_N_' + str(nNodes) + '_' + str(file_id) + '_' + str(nsamples) +'.txt', 'w', bufsize)

    ngraphs = 10 
    graph_dir = "/home/pankaj/Sampling/data/input/social_graphs/N_" + str(nNodes) + "/"

    file_list = os.listdir(graph_dir)
    graph_file_list = []

    for i in range(ngraphs):
        if 'log' not in file_list[i] and 'gt' not in file_list[i]:
            graph_file_list.append(file_list[i])

    G = read_graph(graph_dir + graph_file_list[file_id], nNodes)

    influ_obj = Influence(G, p, num_influ_iter)

    for t in range(4):
        x = torch.rand(nNodes)
        val = []
        tic = time.clock()

        for k in range(20):
            val.append(getRelax(G, x, nsamples, influ_obj, herd = False).item())

        to_write_list = [file_id, np.var(val), (time.clock() - tic)/20]
        print ' '.join(map(str, to_write_list)) + '\n'
        sys.stdout.flush()
        f.write(' '.join(map(str, to_write_list)) + '\n')

def study_k_effect():

    file_id = int(sys.argv[1])
    k = int(sys.argv[2])
    num_influ_iter = 100
    nsample_mlr = 10
    num_fw_iter = 100

    nNodes = 512
    bufsize = 0
    p = 0.4 

    f = open(dirw + 'study_k_effect_p_' + str(p) + '_N_' + str(nNodes) + '_' + str(file_id) + '_' + str(num_influ_iter) + '_' + str(nsample_mlr) + '.txt', 'w', bufsize)

    ngraphs = 10 
    graph_dir = "/home/pankaj/Sampling/data/input/social_graphs/N_" + str(nNodes) + "/"

    file_list = os.listdir(graph_dir)
    graph_file_list = []

    for i in range(ngraphs):
        if 'log' not in file_list[i] and 'gt' not in file_list[i]:
            graph_file_list.append(file_list[i])

    G = read_graph(graph_dir + graph_file_list[file_id], nNodes)

    random_val = select_random_k_pair(G, k, nNodes, 10, p, num_influ_iter) 
    gt_val = get_ground_truth(G, k, nsample_mlr, num_fw_iter, p, num_influ_iter).item()
    to_write_list = [file_id, k, gt_val, random_val]
    print ' '.join(map(str, to_write_list)) + '\n'
    sys.stdout.flush()
    f.write(' '.join(map(str, to_write_list)) + '\n')

def p_study():

    p = float(sys.argv[1])
    nNodes = 16384 
    bufsize = 0
    f = open('variance_study_p_' + str(p) + '_N_' + str(nNodes) + '.txt', 'w', bufsize)

    niter_list = [10, 100, 1000, 1e4]
#    niter_list = [1]
    ngraphs = 2 
    nsamples =  2 

    graph_dir = "/home/pankaj/Sampling/data/input/social_graphs/N_" + str(nNodes) + "/"

    file_list = os.listdir(graph_dir)
    graph_file_list = []

    for i in range(10):
        if 'log' not in file_list[i] and 'gt' not in file_list[i]:
            graph_file_list.append(file_list[i])

    for i in range(ngraphs):
        for iter_num in niter_list:
            G = read_graph(graph_dir + graph_file_list[i], nNodes)
            val = []
            tic = time.clock()
            for j in range(nsamples):
                sample = torch.rand(nNodes) > 0.8
                val.append(ic_model(G, sample, p, iter_num).item())
            to_write_list = [iter_num, np.mean(val), (time.clock() - tic)/nsamples]
            print ' '.join(map(str, to_write_list)) + '\n'
            f.write(' '.join(map(str, to_write_list)) + '\n')

def main():

    nNodes = 1024 
    graph_dir = "/home/pankaj/Sampling/data/input/social_graphs/N_" + str(nNodes) + "/"
    file_list = os.listdir(graph_dir)
    graph_file_list = []

    for i in range(1):
        if 'log' not in file_list[i] and 'gt' not in file_list[i]:
            graph_file_list.append(file_list[i])
    
    p = float(sys.argv[1]) 
    num_influ_iter = 10
    nsample_mlr = int(sys.argv[2]) 
    num_fw_iter = 100

    params = [nNodes, p, num_influ_iter, nsample_mlr, num_fw_iter]
    param_string = '_'.join(str(t) for t in params) 
    bufsize = 0
    f = open(dirw + 'log_k_study_' + param_string + '.txt', 'w', bufsize)

    for i in range(len(graph_file_list)):
        
        filename = graph_dir + graph_file_list[i] 
        results = []

        for k in range(1, 400, 50):
            print i, k
            study_k_effect(filename, k, p, num_influ_iter, nsample_mlr, num_fw_iter, results, i)

        f.write('\n'.join('%s %s %s %s' % x for x in results))
        f.write('\n')
    f.close()

if __name__ == '__main__':
#    influence_variance_study_parallel()
#    multilinear_variance_study() 
    study_k_effect()
