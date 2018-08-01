import sys
import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import csv
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import itertools
from objective import submodObj

def getProb(sample, pVec):
    #sample is 0-1 set and pVec is a probability distribution
    temp = pVec * sample + (1 - pVec)*(1 - sample)
    return torch.prod(temp, 1)

def getLogProb(sample, pVec):
    #sample is 0-1 set and pVec is a probability distribution
    temp = pVec * sample + (1 - pVec)*(1 - sample)
    return torch.sum(torch.log(temp), 1)


def multilinear_importance(x, y, graph_file, nsamples): #x is targe probability vector and y is the proposal distribution
    running_sum = Variable(torch.FloatTensor([0]), requires_grad = True) 
    for trial in range(nsamples):
        sample = Variable(torch.bernoulli(y[0].data))
        targetP = getProb(sample, x)
        proposalP = getProb(sample, y)
        assert proposalP.data[0] > 0, "Proposal distribution invalid"
        running_sum = running_sum + submodObj(graph_file, sample)*(float(targetP)/proposalP)
    mc_val = running_sum/nsamples
    return mc_val.data[0]


def variance_estimate(input, proposal, graph_file, nsamples):
    variance_val = []
    batch_size = int(input.size()[0])
    
#    N = int(np.sqrt(int(L_mat[0].shape[0])))
    N = int(input.shape[1])

    for instance in range(batch_size):
        fval = []
        for t in range(50): #50 seems to work well in practice - smaller (say 20) leads to less consistency of variance
            x = input[instance].unsqueeze(0)
            y = proposal[instance].unsqueeze(0)
            temp = multilinear_importance(x, y, graph_file, nsamples)
            fval.append(temp)
        variance_val.append(np.std(fval)**2)
    return np.median(variance_val), np.mean(variance_val) 

def get_variance(N, g_id, k, nsamples, num_fw_iter, p, num_influ_iter, if_herd, a):

    graph_file = '/home/pankaj/Sampling/data/input/social_graphs/N_' + str(N) + '/graphs/g_N_' + str(N) + '_' + str(g_id) + '.txt'

    G = read_graph(graph_file, N)

    x_good_sfo = get_sfo_optimum('/home/pankaj/Sampling/data/input/social_graphs/N_' + str(N) + '/sfo_gt/g_N_' + str(N) + '_id_' + str(g_id) + '_k_' + str(k) + '.txt', N) 

    x_good_fw = get_fw_optimum('/home/pankaj/Sampling/data/input/social_graphs/N_' + str(N) + '/fw_gt/g_N_' + str(N) + '_id_' + str(g_id) + '_k_' + str(k) + '_100.txt', N) 

    temp = '/home/pankaj/Sampling/data/input/social_graphs/N_' + str(N) + '/var_study/g_N_' + str(N) + '_' + str(g_id) 

    var_file = '_'.join(str(x) for x in [temp, k, nsamples, num_fw_iter, p, num_influ_iter, if_herd, a]) + '.txt'

    variance_study(G, nsamples, k, var_file, num_fw_iter, p, num_influ_iter, if_herd, x_good_sfo, x_good_fw, a) 

def variance_study(G, nsamples, k, var_file, num_fw_iter, p, num_influ_iter,
        if_herd, x_good_sfo, x_good_fw, a):

    N = nx.number_of_nodes(G)

    influ_obj = Influence(G, p, num_influ_iter)

    x = torch.Tensor([0.5]*N) 

    temp = []

    for t in range(20):
        val1 = getImportanceRelax(G, x_good_sfo, x, nsamples, influ_obj, if_herd, a).item()
        val2 = getImportanceRelax(G, x_good_fw, x, nsamples, influ_obj, if_herd, a).item()
        val3 = getRelax(G, x, nsamples, influ_obj, if_herd).item()
        print(val1, val2, val3)
        temp.append((val1, val2, val3))

    print('\n'*2)
    print("sfo var= ", np.std([t[0] for t in temp]), "  mean = ", np.mean([t[0] for t in temp]))
    print("fw var = ", np.std([t[1] for t in temp]), "  mean = ", np.mean([t[1] for t in temp]))
    print("mc var = ", np.std([t[2] for t in temp]), "  mean = ", np.mean([t[2] for t in temp]))
    print("true relax value = ", getRelax(G, x, 100, influ_obj, if_herd).item())

    f = open(var_file, 'w', 0)
    f.write(str(np.std([t[0] for t in temp]))+ " " + str(np.mean([t[0] for t in
        temp])) + "\n")
    f.write(str(np.std([t[1] for t in temp]))+ " " + str(np.mean([t[1] for t in
        temp])) + "\n")
    f.write(str(np.std([t[2] for t in temp]))+ " " + str(np.mean([t[2] for t in
        temp])) + "\n")
    f.write(str(getRelax(G, x, 100, influ_obj, if_herd).item()) + "\n")

    f.close()


