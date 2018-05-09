import torch
import networkx as nx
import sys
import numpy as np
import torch.optim as optim
import math

from graphnet import GraphConv, GraphScorer, MyNet
from torch.autograd import Variable
from objective import submodObj
from variance import variance_estimate

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

#filename = ['/home/pankaj/Sampling/data/working/06_05_2018/net1-network.txt','/home/pankaj/Sampling/data/working/06_05_2018/net2-network.txt'] 
filename = '/home/pankaj/Sampling/data/working/08_05_2018/128_n_100_e_net-network.txt'
batch_size = 3 
eps = 1e-6 #for numerical stability in weight matrix - required for centrality computations
#filename = './influmax/example-network.txt'
f = open(filename, 'rU')
nNodes = 0
for line in f:
    if line == '\n':
        break
    else:
        nNodes += 1

alphas = np.zeros((batch_size, nNodes, nNodes))
adjacency = np.zeros((batch_size, nNodes, nNodes))

for line in f:
    temp = line.strip('\n')
    temp2 = temp.split(",")
    a = int(temp2[0])
    b = int(temp2[1])
    w_ab = float(temp2[2])
    alphas[0, a, b] = w_ab
    if w_ab == 0:
        adjacency[0, a, b] = 0
    else:
        adjacency[0, a, b] = 1

max_alpha = np.max(alphas)
#alphas are affinity measures - take -ve to get a dissimilarity measure
weights = max_alpha - alphas + eps
weights = Variable(torch.from_numpy(weights)).float()
adjacency = Variable(torch.from_numpy(adjacency)).float()
w = np.asmatrix(weights.data[0])
G = nx.DiGraph(w)

w = torch.from_numpy(w.astype("float32")).clone()

N = nx.number_of_nodes(G)
x = Variable((torch.rand(batch_size, N)).float(), requires_grad = True)

#Node features
#I'm assuming that the dictionary is sorted by keys (node ids)
in_degree = np.array((nx.in_degree_centrality(G)).values())
in_degree = Variable(torch.from_numpy(in_degree).clone().repeat(batch_size, 1))

out_degree = np.array((nx.out_degree_centrality(G)).values())
out_degree = Variable(torch.from_numpy(out_degree).clone().repeat(batch_size, 1))

closeness = np.array((nx.closeness_centrality(G)).values())
closeness = Variable(torch.from_numpy(closeness).clone().repeat(batch_size, 1))

between = np.array((nx.betweenness_centrality(G)).values())
between = Variable(torch.from_numpy(between).clone().repeat(batch_size, 1))

eigen_central = np.array((nx.eigenvector_centrality(G)).values())
eigen_central = Variable(torch.from_numpy(eigen_central).clone().repeat(batch_size, 1))

pagerank = np.array((nx.pagerank(G)).values())
pagerank = Variable(torch.from_numpy(pagerank).clone().repeat(batch_size, 1))

to_stack = [in_degree, out_degree, closeness, between, eigen_central, pagerank]

node_feat = torch.stack(to_stack, 2).float()

net = MyNet()

lr1 = 1e-2

optimizer = optim.Adam(net.parameters(), lr=lr1)

for epoch in range(1):
    #get minibatch
    optimizer.zero_grad()   # zero the gradient buffers
    output = net(x, adjacency, weights, node_feat) 
    loss = reconstruction_loss(x, output)
    print "Epoch: ", epoch, "       loss (l2 reconstruction) = ", loss.data[0]
    loss.backward()
    optimizer.step()    # Does the update

lr2 = 1e-4
optimizer2 = optim.Adam(net.parameters(), lr=lr2)

num_samples_mc = 2

#build cache of submodular func. evals of sets from x
cache = []
for t in x:
    for p in range(num_samples_mc):
        #draw set from x
        sample = Variable(torch.bernoulli(t.squeeze(0).data))
        val = submodObj(filename, sample)
        cache.append((t, sample, val)) 

print cache[-1]
sys.exit()

for epoch in range(1):
    optimizer2.zero_grad()   # zero the gradient buffers
    output = net(x, adjacency, weights, node_feat) 
    loss = kl_loss_mc(x, output, filename, num_samples_mc)
    print "Epoch: ", epoch, "       loss = ", loss.data[0]
    loss.backward()
    optimizer2.step()    # Does the update

sys.exit()

sample_list = [int(math.pow(2, t)) for t in range(1, N - 2)]
 

mu = net(x, adjacency, weights, None) 
output = scorer(mu)

for nsample in sample_list:
    print "#samples = ", nsample, " ", variance_estimate(x, output, filename, nsample)
#print mu
