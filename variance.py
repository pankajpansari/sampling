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
