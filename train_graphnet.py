import torch
import networkx as nx
import sys
import numpy as np
import torch.optim as optim
import math

from graphnet import GraphConv, GraphScorer, MyNet
from torch.autograd import Variable
from influence import ic_model as submodObj
from variance import variance_estimate
import matplotlib.pyplot as plt

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

filename = ['/home/pankaj/Sampling/data/input/social_graphs/k_5/g_k_5_999-network.txt', '/home/pankaj/Sampling/data/input/social_graphs/k_5/g_k_5_998-network.txt']
batch_size = len(filename) 
eps = 1e-6 #for numerical stability in weight matrix - required for centrality computations

k = 5
adjacency = Variable(torch.zeros(batch_size, int(math.pow(2, k)), int(math.pow(2, k))))

for this_graph in range(batch_size):
    f = open(filename[this_graph], 'rU')

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


    adjacency[this_graph] = Variable(torch.from_numpy(this_graph_adj)).float()

#for this_graph in range(batch_size):
#    G = nx.DiGraph(adjacency[this_graph].numpy())
#
#    N = nx.number_of_nodes(G)
#    E = nx.number_of_edges(G)
#
#    #Query stats to check if graph read properly
#    print N, E
#    for t in range(N):
#        print t, ":", 
#        for p in G.neighbors(t):
#            print p, 
#        print
#
#    plt.imshow(adjacency[this_graph].numpy())
#    plt.show()

#sys.exit()
#filename = './influmax/example-network.txt'


#Node features
#I'm assuming that the dictionary is sorted by keys (node ids)

num_node_feat = 6
node_feat = Variable(torch.zeros(batch_size, int(math.pow(2, k)), num_node_feat))

for this_graph in range(batch_size):

    G = nx.DiGraph(adjacency[this_graph].numpy())

    in_degree = np.array((nx.in_degree_centrality(G)).values())

    out_degree = np.array((nx.out_degree_centrality(G)).values())

    closeness = np.array((nx.closeness_centrality(G)).values())

    between = np.array((nx.betweenness_centrality(G)).values())

    eigen_central = np.array((nx.eigenvector_centrality(G)).values())

    pagerank = np.array((nx.pagerank(G)).values())

    to_stack = [in_degree, out_degree, closeness, between, eigen_central, pagerank]

    assert(len(to_stack) == num_node_feat)

    node_feat[this_graph] = Variable(torch.from_numpy(np.stack(to_stack, 1)).float())

net = MyNet()
print net(adjacency, node_feat)
sys.exit()

lr1 = 1e-2

optimizer = optim.Adam(net.parameters(), lr=lr1)

for epoch in range(10):
    #get minibatch
    optimizer.zero_grad()   # zero the gradient buffers
    output = net(x, adjacency, weights, node_feat) 
    loss = reconstruction_loss(x, output)
    print "Epoch: ", epoch, "       loss (l2 reconstruction) = ", loss.item()
    loss.backward()
    optimizer.step()    # Does the update
