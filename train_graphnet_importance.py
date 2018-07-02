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
torch.set_printoptions(precision = 4)
torch.manual_seed(123)

def reconstruction_loss(x, y):
    #Reconstruction loss - L2 difference between x and y 
    batch_size = x.size()[0]
    temp = x - y
    l2_norms = torch.norm(temp, 2, 1)
    return ((l2_norms**2).sum())/batch_size

def getProb(sample, pVec):
    #sample is 0-1 set and pVec is a probability distribution
    temp = pVec * sample + (1 - pVec)*(1 - sample)
    return torch.prod(temp, 1)

def entropy_mc(output, nsamples):
    batch_size = output.size()[0]
    ent_sum = torch.zeros(batch_size) 
    for t in range(nsamples):
        sample = torch.bernoulli(output)
        ent_sum += -torch.log(getProb(sample, output))
    return ent_sum.sum()/(nsamples*batch_size)

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
            if yP.item() == 0:
                print xP, yP
                print x_t, y_t, sample
                sys.exit()
            term1 =  torch.log(yP)
            term2 =  torch.log(torch.abs(val)*xP)
            term3 = yP/xP
            obj_t += term3*(term1 - term2) 
        obj[t] = obj_t/nsamples
    return obj.mean()

def kl_loss_mc_y(x, y, nsamples, influ_obj):
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
            sample = Variable(torch.bernoulli(y_t.squeeze().data))
            val = torch.abs(influ_obj(sample.numpy()))
            xP = getProb(sample, x_t)
            yP = getProb(sample, y_t)   
            if xP.item() == 0:
                print xP, yP
                print x_t, y_t, sample
                sys.exit()
            term1 =  torch.log(yP)
            term2 =  torch.log(torch.abs(val)*xP)
            obj_t += term1 - term2
        obj[t] = obj_t/nsamples
    return obj.mean()

def kl_loss_mc_uniform(x, y, nsamples, influ_obj):
    #Sampling from x distribution
    batch_size = x.size()[0]
    obj = Variable(torch.FloatTensor([0]*batch_size)) 
    N = x.size()[1]

    assert(x.dim() == 2)
    assert(y.dim() == 2)

    uniformP = Variable(torch.FloatTensor([1.0/math.pow(2, N)]))

    for t in range(batch_size):
        y_t = y[t, :].unsqueeze(0)
        x_t = x[t, :].unsqueeze(0)
        obj_t = Variable(torch.FloatTensor([0])) 
        for p in range(nsamples):
            #draw a sample/set from the uniform distribution
            sample = Variable(torch.bernoulli(torch.FloatTensor([0.5]*N)))
            val = torch.abs(influ_obj(sample.numpy()))
            xP = getProb(sample, x_t)
            yP = getProb(sample, y_t)   
            if yP.item() == 0:
                print xP, yP
                print x_t, y_t, sample
                sys.exit()
            term1 =  torch.log(yP)
            term2 =  torch.log(torch.abs(val)*xP)
            term3 = yP/uniformP
            obj_t += term3*(term1 - term2) 
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

    lr1 = float(sys.argv[1])
    lr2 = float(sys.argv[2])
    mom = float(sys.argv[3])
    n_epochs1 = int(sys.argv[4])
    n_epochs2 = int(sys.argv[5])
    batch_size = int(sys.argv[6])
    p = 0.5
    num_influ_iter = 100

    graph_dir = '/home/pankaj/Sampling/data/input/social_graphs/N_32/'
    data_list = ['g_N_32_99.txt']

    nNodes = 32 
    k = 9 #has no effect as of now
    net = MyNet(k)

    x = Variable(torch.rand(batch_size, nNodes), requires_grad = True) 
#    y = Variable(torch.rand(batch_size, nNodes), requires_grad = True) 
    temp = get_adjacency(data_list, nNodes, graph_dir)

    #In this case, repeat adj batch_size number of times
    adj = temp.repeat(batch_size, 1, 1)

    node_feat = get_node_feat(adj, nNodes)

    for params in net.parameters():
        params.requires_grad = True


    optimizer = optim.Adam(net.parameters(), lr=lr1)
#    optimizer = optim.SGD(net.parameters(), lr=lr)

#    for epoch in range(n_epochs1):
    for epoch in range(0):

        #get minibatch
        optimizer.zero_grad()   # zero the gradient buffers
    
        y = net(x, adj, node_feat) 
 
        train_loss = reconstruction_loss(x, y)

        print "Epoch: ", epoch, "       reconstruction loss = ", train_loss.item()

        train_loss.backward()
    
        optimizer.step()    # Does the update

#    torch.save(net, '/home/pankaj/Sampling/data/working/02_07_2018_temp.net')
    net = torch.load('/home/pankaj/Sampling/data/working/02_07_2018_temp.net')

    G = nx.DiGraph(adj[0].numpy())

    influ_obj = Influence(G, p, num_influ_iter)

    optimizer = optim.SGD(net.parameters(), lr=lr2)
#    optimizer = optim.Adam(net.parameters(), lr=lr2)

    for epoch in range(n_epochs2):

        #get minibatch
        optimizer.zero_grad()   # zero the gradient buffers
    
        y = net(x, adj, node_feat) 
 
#        train_loss = kl_loss_mc_uniform(x, y, 10, influ_obj)
        train_loss2 = kl_loss_mc_y(x, y, 100, influ_obj)

        print "Epoch: ", epoch, "       KL-based loss = ", train_loss2.item()

        train_loss2.backward()
    
        optimizer.step()    # Does the update
    
    sys.exit()
 
if __name__ == '__main__':
    main()
