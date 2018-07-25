import networkx as nx
import sys
import numpy as np
import math
import torch
from influence import ic_model as submodObj
from influence import Influence 
from torch.autograd import Variable
import logger
from builtins import range
import time

def herd_points(probs, num):
    """ Based on Welling & Chen (2010), eqn (18) and (19) """
    x = probs > 0.5
    w = probs - x[-1].float()
    x = x.unsqueeze(0)

    for i in range(num - 1):
        x_next = (w > 0.5)
        w = w + probs - x_next.float() # np.mean(x, 0) - x_next 
        x = torch.cat((x, x_next.unsqueeze(0)))

    return x.float()

def getRelax(G, x, nsamples, influ_obj, herd): 

    current_sum = Variable(torch.FloatTensor([0]), requires_grad = False) 

    if herd == 1:
        samples_list = herd_points(x, nsamples) 
    else:
        samples_list = Variable(torch.bernoulli(x.repeat(nsamples, 1)))

    for sample in samples_list:
        current_sum = current_sum + influ_obj(sample.numpy())

    return current_sum/nsamples

def getCondGrad(grad, k):

    #conditional gradient for cardinality constraints

    N = grad.shape[0]
    top_k = Variable(torch.zeros(N)) #conditional grad
    sorted_ind = torch.sort(grad, descending = True)[1][0:k]
    top_k[sorted_ind] = 1
#    positive = grad > 0
#    top_k_positive = (top_k.byte() * positive).float()
#    return top_k_positive 
    return top_k

def getGrad(G, x, nsamples, influ_obj, herd):

    #Returns the gradient vector of the multilinear relaxation at x as given in Chekuri's paper
    #(See Theorem 1 in nips2012 paper)

    N = G.number_of_nodes()
    grad = Variable(torch.zeros(N))

    if herd == 1: 
        samples_list = herd_points(x, nsamples) 
    else:
        samples_list = Variable(torch.bernoulli(x.repeat(nsamples, 1)))

    for t in range(nsamples):
        sample = samples_list[t] 
        m = torch.zeros(sample.size()) 
        for p in np.arange(N):
            m[p] = 1
            grad[p] = grad[p] + (influ_obj(np.logical_or(sample.numpy(), m.numpy())) - influ_obj(np.logical_and(sample.numpy(), np.logical_not(m.numpy()))))
            m[p] = 0
    return grad*1.0/nsamples


def runFrankWolfe(G, nsamples, k, file_prefix, num_fw_iter, p, num_influ_iter, if_herd):

    N = nx.number_of_nodes(G)

    x = Variable(torch.Tensor([1.0*k/N]*N))
    
    bufsize = 0
    f = open(file_prefix + '_log.txt', 'w', bufsize)

    influ_obj = Influence(G, p, num_influ_iter)

    tic = time.clock()

    iter_num = 0
    obj = getRelax(G, x, nsamples, influ_obj, if_herd)
    toc = time.clock()

    print "Iteration: ", iter_num, "    obj = ", obj.item(), "  time = ", (toc - tic),  "   Total/New/Cache: ", influ_obj.itr_total , influ_obj.itr_new , influ_obj.itr_cache

    f.write(str(toc - tic) + " " + str(obj.item()) + " " + str(influ_obj.itr_total) + '/' + str(influ_obj.itr_new) + '/' + str(influ_obj.itr_cache) + "\n") 

    for iter_num in np.arange(1, num_fw_iter):

        influ_obj.counter_reset()

        grad = getGrad(G, x, nsamples, influ_obj, if_herd)

        x_star = getCondGrad(grad, k)

        step = 2.0/(iter_num + 2) 

        x = step*x_star + (1 - step)*x

        obj = getRelax(G, x, nsamples, influ_obj, if_herd)
        
        toc = time.clock()

        print "Iteration: ", iter_num, "    obj = ", obj.item(), "  time = ", (toc - tic),  "   Total/New/Cache: ", influ_obj.itr_total , influ_obj.itr_new , influ_obj.itr_cache

        f.write(str(toc - tic) + " " + str(obj.item()) + " " + str(influ_obj.itr_total) + '/' + str(influ_obj.itr_new) + '/' + str(influ_obj.itr_cache) + "\n") 

    f.close()

    return x

def main():
    grad = torch.randn(10)

if __name__ == '__main__':
    main()
