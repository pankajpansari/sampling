import networkx as nx
import sys
import numpy as np
import math
import torch
from influence import ic_model as submodObj
from torch.autograd import Variable
import logger
from builtins import range

def getRelax(G, x, nsamples): 

    current_sum = Variable(torch.FloatTensor([0]), requires_grad = True) 

    for trial in range(nsamples):
        sample = Variable(torch.bernoulli(x.data))
        current_sum = current_sum + submodObj(G, sample)

    return current_sum/nsamples

def getCondGrad(grad, k):
    #conditional gradient for cardinality constraints
    N = grad.shape[0]
    top_k = Variable(torch.zeros(N)) #conditional grad
    sorted_ind = torch.sort(grad, descending = True)[1][0:k]
    top_k[sorted_ind] = 1
    positive = grad > 0
    top_k_positive = (top_k.byte() * positive).float()
    return top_k_positive 

def getGrad(G, x, nsamples):
#Returns the gradient vector of the multilinear relaxation at x as given in Chekuri's paper
#(See Theorem 1 in nips2012 paper)
    N = G.number_of_nodes()
    grad = Variable(torch.zeros(N))

    for p in np.arange(N):
        x_include = x.clone()
        x_exclude  = x.clone() 
        x_include[p] = 1
        x_exclude[p] = 0

        grad[p] =  getRelax(G, x_include, nsamples) - getRelax(G, x_exclude, nsamples)
    return grad


def runFrankWolfe(G, nsamples):

    N = nx.number_of_nodes(G)

    x = Variable(torch.Tensor([0]*N))
    
    for iter_num in np.arange(1, 50):

        grad = getGrad(G, x, nsamples)

        x_star = getCondGrad(grad, 10)

        step = 2.0/(iter_num + 2) 

        x = step*x_star + (1 - step)*x

        obj = getRelax(G, x, nsamples)
        
        print "Iteration: ", iter_num, "    obj = ", obj, "     x = ", x
   return x 

def main():
    grad = torch.randn(10)

if __name__ == '__main__':
    main()
