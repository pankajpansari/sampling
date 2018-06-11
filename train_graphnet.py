import torch
import networkx as nx
import sys
import numpy as np
import torch.optim as optim
import math
import os
import logger
import random

from graphnet import GraphConv, GraphScorer, MyNet
from torch.autograd import Variable
from influence import ic_model as submodObj
from variance import variance_estimate
import matplotlib.pyplot as plt
import visdom

def getProb(sample, pVec):
    #sample is 0-1 set and pVec is a probability distribution
    temp = pVec * sample + (1 - pVec)*(1 - sample)
    return torch.prod(temp, 1)

def getLogProb(sample, pVec):
    #sample is 0-1 set and pVec is a probability distribution
    temp = pVec * sample + (1 - pVec)*(1 - sample)
    return torch.sum(torch.log(temp), 1)


def kl_loss_mc(input, proposal, graph_file, nsamples):
    #Estimate the objective function using sets from uniform distribution
    batch_size = input.size()[0]
    obj = Variable(torch.FloatTensor([0]*batch_size)) 
    N = int(input.shape[1])

    for t in range(nsamples):
        #draw a sample/set from the uniform distribution
#        sample = Variable(torch.bernoulli(torch.FloatTensor([0.5]*N)))
        sample = Variable(torch.bernoulli(input.squeeze(0).data))
        inputlogP = getLogProb(sample, input)
        proplogP = getLogProb(sample, proposal)
        propP = getProb(sample, proposal)
        inputP = getProb(sample, input)
        print sample, inputP, propP, val
        val = submodObj(graph_file, sample) 
#        obj = torch.add(obj, (propP/uniformP) *(proplogP - (inputlogP + torch.log(val))))
#        print sample.data, inputP.data, propP.data, (propP/inputP).data 
#        print (propP/inputP).data[0] , (proplogP - inputlogP).data[0], torch.log(val).data[0]
        obj = torch.add(obj, (propP/inputP) *(proplogP - (inputlogP + torch.log(val))))
    return obj.mean()/nsamples


def reconstruction_loss(input, proposal):
    #Reconstruction loss - L2 difference between input and proposal 
    batch_size = input.size()[0]
    temp = input - proposal
    l2_norms = torch.norm(temp, 2, 1)
    return ((l2_norms**2).sum())/batch_size

graph_dir = '/home/pankaj/Sampling/data/input/social_graphs/k_5/'
file_list = os.listdir(graph_dir)
network_file_list = []
for this_file in file_list:
    if 'network' in this_file:
        network_file_list.append(graph_dir + this_file)

#network_file_list = ['/home/pankaj/Sampling/data/input/social_graphs/k_5/g_k_5_999-network.txt', '/home/pankaj/Sampling/data/input/social_graphs/k_5/g_k_5_998-network.txt']
#batch_size = len(network_file_list) 
def get_data(start_id, end_id):
    batch_size = end_id - start_id 
    
    k = 5
    adjacency = Variable(torch.zeros(batch_size, int(math.pow(2, k)), int(math.pow(2, k))))
    ground_truth = Variable(torch.zeros(batch_size, int(math.pow(2, k))))
    
    for this_graph in range(batch_size):
    
        f = open(network_file_list[start_id + this_graph], 'rU')
    
        nNodes = 0
        for line in f:
            if line == '\n':
                break
            else:
                nNodes += 1
    
        this_graph_adj = np.zeros((nNodes, nNodes))
    
        for line in f:
            temp = line.strip('\n')
            temp2 = temp.split(",")
            a = int(temp2[0])
            b = int(temp2[1])
            w_ab = float(temp2[2])
            if w_ab == 0:
                this_graph_adj[a, b] = 0
            else:
                this_graph_adj[a, b] = 1
    
        f.close()
        adjacency[this_graph] = Variable(torch.from_numpy(this_graph_adj)).float()
    
        gt_filename = network_file_list[this_graph].replace('-network.txt', '_gt.txt')
    
        f2 = open(gt_filename, 'rU')
    
        count = 0
        for line in f2:
            ground_truth[this_graph][count] = float(line)
            count += 1
    
        f2.close()
    
    num_node_feat = 5
    node_feat = Variable(torch.zeros(batch_size, int(math.pow(2, k)), num_node_feat))
    
    for this_graph in range(batch_size):
    
    #    print network_file_list[this_graph]
        G = nx.DiGraph(adjacency[this_graph].numpy())
    
        in_degree = np.array((nx.in_degree_centrality(G)).values())
    
        out_degree = np.array((nx.out_degree_centrality(G)).values())
    
        closeness = np.array((nx.closeness_centrality(G)).values())
    
        between = np.array((nx.betweenness_centrality(G)).values())
    
#        eigen_central = np.array((nx.eigenvector_centrality(G)).values())
    
        pagerank = np.array((nx.pagerank(G)).values())
    
#        to_stack = [in_degree, out_degree, closeness, between, eigen_central, pagerank]
        to_stack = [in_degree, out_degree, closeness, between, pagerank]
    
        assert(len(to_stack) == num_node_feat)
    
        node_feat[this_graph] = Variable(torch.from_numpy(np.stack(to_stack, 1)).float())

    return adjacency, node_feat, ground_truth

net = MyNet()

lr = 1e-3
n_epochs = 50

#----------------------------------------------------------
# Prepare logging
#----------------------------------------------------------

# create Experiment
xp = logger.Experiment("train_graphnet", use_visdom=True, visdom_opts={'server': 'http://localhost', 'port': 8097}, time_indexing=False, xlabel='Epoch')

# log the hyperparameters of the experiment
xp.log_config({'lr': lr, 'n_epochs': n_epochs})

# create parent metric for training metrics (easier interface)
#xp.ParentWrapper(tag='train', name='parent', children=(xp.AvgMetric(name='loss')))
xp.ParentWrapper(tag='train', name='parent', children=(xp.SimpleMetric(name='loss'),))
xp.ParentWrapper(tag='val', name='parent', children=(xp.SimpleMetric(name='loss'),))
optimizer = optim.Adam(net.parameters(), lr=lr)

log_prefix = '/home/pankaj/Sampling/data/working/05_06_2018/'
f = open(log_prefix + 'training_log.txt', 'w')

tr_adj, tr_node_feat, tr_gt = get_data(0, 50)
print "Got training data"
val_adj, val_node_feat, val_gt = get_data(50, 100)
print "Got validation data"

for epoch in range(n_epochs):
    #get minibatch
    optimizer.zero_grad()   # zero the gradient buffers

    tr_output = net(tr_adj, tr_node_feat) 
    tr_loss = reconstruction_loss(tr_gt, tr_output)

    val_output = net(val_adj, val_node_feat) 
    val_loss = reconstruction_loss(val_gt, val_output)

    print "Epoch: ", epoch, "       train loss = ", tr_loss.item(), "      val loss = ", val_loss.item()

    tr_loss.backward()
    val_loss.backward()

    optimizer.step()    # Does the update

    xp.Parent_Train.update(loss=tr_loss)
    xp.Parent_Train.log()

    xp.Parent_Val.update(loss=val_loss)
    xp.Parent_Val.log()

#Query some input, output values
tr_output = net(tr_adj, tr_node_feat) 
val_output = net(val_adj, val_node_feat) 

print '-'*50
for t in range(2):
    rand_ind = random.randrange(0, 50)
    print tr_gt[rand_ind], tr_output[rand_ind]  
    print reconstruction_loss(tr_gt[rand_ind].unsqueeze(0), tr_output[rand_ind].unsqueeze(0))
    print tr_node_feat[t]
xp.to_json(log_prefix + "train_log.json")

f.close()
