import torch
import networkx as nx
import sys

from graphnet import GraphConv, GraphScorer
from torch.autograd import Variable

N = 10
n_layer = 3
p = 32
w_std = 0.1
# Create a sample
g = nx.erdos_renyi_graph(n = N, p = 0.15)
A_t = nx.adjacency_matrix(g).todense()
A_t = torch.from_numpy(A_t.astype("float32")).clone()


x_t = (torch.randn(N) >= 0).float()
w_t = A_t.clone()

# batch 2
x = Variable(x_t.unsqueeze(0).repeat(2, 1))
A = Variable(A_t.unsqueeze(0).repeat(2, 1, 1))
w = Variable(w_t.unsqueeze(0).repeat(2, 1, 1))

net = GraphConv(n_layer, p, w_std)
scorer = GraphScorer(p, w_std)

mu = net(x, A, w, None)

scores = scorer(mu)
print scores
#print mu
