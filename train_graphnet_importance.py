import torch
import networkx as nx
import sys
import numpy as np
import torch.optim as optim
import math
import os
import logger
import random
import fnmatch
import time
import matplotlib.pyplot as plt

from influence import Influence 

from graphnet import GraphConv, GraphScorer, MyNet
from torch.autograd import Variable
from variance import variance_estimate
from train_graphnet import get_node_feat

random.seed(123)
np.random.seed(123)
torch.set_printoptions(precision = 2)
torch.manual_seed(123)

def getProb(sample, pVec):
    #sample is 0-1 set and pVec is a probability distribution
    temp = pVec * sample + (1 - pVec)*(1 - sample)
    return torch.prod(temp, 1)

def kl_loss_mc_x(x, y, nsamples, influ_obj):
    #Sampling from x distribution
    batch_size = x.size()[0]
    obj = Variable(torch.FloatTensor([0]*batch_size)) 

    assert(x.dim() == 2)
    assert(y.dim() == 2)

    for t in range(batch_size):
        y_t = y[t, :].unsqueeze(0)
        x_t = x[t, :].unsqueeze(0)
        obj_t = Variable(torch.FloatTensor([0])) 
        for p in range(nsamples):
            #draw a sample/set from the uniform distribution
            sample = Variable(torch.bernoulli(x_t.squeeze().data))
            val = torch.abs(influ_obj(sample.numpy()))
            xP = getProb(sample, x_t)
            yP = getProb(sample, y_t)
            obj_t += (yP/xP)*(torch.log(yP) - torch.log(torch.abs(val)*xP))
        obj[t] = obj_t/nsamples
    return obj.mean()


def get_adjacency(data_list, nNodes, graph_dir):

    adjacency = Variable(torch.zeros(len(data_list), nNodes, nNodes))

    for t in range(len(data_list)):
        f = open(graph_dir + data_list[t], 'rU')
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

        adjacency[t] = Variable(torch.from_numpy(adj)).float()

    return adjacency


def main():

    lr = float(sys.argv[1])
    mom = float(sys.argv[2])
    n_epochs = int(sys.argv[3])
    batch_size = int(sys.argv[4])
    p = 0.5
    num_influ_iter = 100

    graph_dir = '/home/pankaj/Sampling/data/input/social_graphs/N_32/'
    data_list = ['g_N_32_99.txt']

    nNodes = 32 
    k = 9 #has no effect as of now
    net = MyNet(k)

    x = Variable(torch.rand(batch_size, nNodes), requires_grad = True) 
    temp = get_adjacency(data_list, nNodes, graph_dir)

    #In this case, repeat adj batch_size number of times
    adj = temp.repeat(batch_size, 1, 1)

    node_feat = get_node_feat(adj, nNodes)

    for params in net.parameters():
        params.requires_grad = True

    loss = kl_loss_mc_x

    optimizer = optim.Adam(net.parameters(), lr=lr)
   
    G = nx.DiGraph(adj[0].numpy())

    influ_obj = Influence(G, p, num_influ_iter)

    for epoch in range(n_epochs):

        #get minibatch
        optimizer.zero_grad()   # zero the gradient buffers
    
        y = net(x, adj, node_feat) 
    
        train_loss = loss(x, y, 100, influ_obj)
 
        print "Epoch: ", epoch, "       train loss = ", train_loss.item()

        train_loss.backward()
    
        optimizer.step()    # Does the update

if __name__ == '__main__':
    main()
