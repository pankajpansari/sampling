import networkx as nx
import sys
import numpy as np
import math
import torch
from influence import ic_model as submodObj
from influence import Influence 
from torch.autograd import Variable
from frank_wolfe import getRelax
#from __future__ import print_function
import logger
from builtins import range
import time
np.random.seed(1234)
torch.manual_seed(1234) 

def getProb(sample, pVec):
    #sample is 0-1 set and pVec is a probability distribution
    temp = pVec * sample + (1 - pVec)*(1 - sample)
    return torch.prod(temp, 1)

def getLogProb(sample, pVec):
    #sample is 0-1 set and pVec is a probability distribution
    a = torch.log(pVec)
    b = torch.log(1 - pVec)
    logp = a * sample + b * (1 - sample)
    return logp.sum(1)

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

def getImportanceWeights(samples_list, nominal, proposal):
    logp_nom = getLogProb(samples_list, nominal)
    logp_prp = getLogProb(samples_list, proposal)
    return torch.exp(logp_nom - logp_prp)

def getImportanceRelax(G, x_good, x, nsamples, influ_obj, herd, a): 

    current_sum = Variable(torch.FloatTensor([0]), requires_grad = False) 

    x_prp = (1 - a)*x + a*x_good

    if herd == 1:
        samples_list = herd_points(x_prp, nsamples) 
    else:
        samples_list = Variable(torch.bernoulli(x_prp.repeat(nsamples, 1)))

    w = getImportanceWeights(samples_list, x, x_prp)

    for i in range(nsamples):
        current_sum = current_sum + w[i]*influ_obj(samples_list[i].numpy())

    return current_sum/nsamples

def getCondGrad(grad, k):

    #conditional gradient for cardinality constraints

    N = grad.shape[0]
    top_k = Variable(torch.zeros(N)) #conditional grad
    sorted_ind = torch.sort(grad, descending = True)[1][0:k]
    top_k[sorted_ind] = 1
    return top_k


def getImportanceGrad(G, x_good, x, nsamples, influ_obj, herd, a):

    #Returns the gradient vector of the multilinear relaxation at x as given in Chekuri's paper
    #(See Theorem 1 in nips2012 paper)

    N = G.number_of_nodes()
    grad = Variable(torch.zeros(N))

    x_prp = (1 - a)*x + a*x_good

    if herd == 1: 
        samples_list = herd_points(x_prp, nsamples) 
    else:
        samples_list = Variable(torch.bernoulli(x_prp.repeat(nsamples, 1)))

    w = getImportanceWeights(samples_list, x, x_prp)

    for t in range(nsamples):
        sample = samples_list[t] 
        m = torch.zeros(sample.size()) 
        for p in np.arange(N):
            m[p] = 1
            grad[p] = grad[p] + w[t]*(influ_obj(np.logical_or(sample.numpy(), m.numpy())) - influ_obj(np.logical_and(sample.numpy(), np.logical_not(m.numpy()))))
            m[p] = 0

    return grad*1.0/nsamples


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

def runImportanceFrankWolfe(G, nsamples, k, log_file, opt_file, num_fw_iter, p, num_influ_iter, if_herd, x_good, a):

    N = nx.number_of_nodes(G)

    influ_obj = Influence(G, p, num_influ_iter)

    x = Variable(torch.Tensor([1.0*k/N]*N))

    bufsize = 0

    f = open(log_file, 'w', bufsize)

    tic = time.clock()

    iter_num = 0
    obj = getImportanceRelax(G, x_good, x, nsamples, influ_obj, if_herd, a)
    toc = time.clock()

    influ_val = []
    influ_val_best = []
    influ_best = -10

    print "Iteration: ", iter_num, "    obj = ", obj.item(), "  time = ", (toc - tic),  "   Total/New/Cache: ", influ_obj.itr_total , influ_obj.itr_new , influ_obj.itr_cache

    f.write(str(toc - tic) + " " + str(obj.item()) + " " + str(influ_obj.itr_total) + '/' + str(influ_obj.itr_new) + '/' + str(influ_obj.itr_cache) + "\n") 

    for iter_num in np.arange(1, num_fw_iter):

        influ_obj.counter_reset()

        grad = getImportanceGrad(G, x_good, x,nsamples, influ_obj, if_herd, a)

        x_star = getCondGrad(grad, k)

        step = 2.0/(iter_num + 2) 

        x = step*x_star + (1 - step)*x

        obj = getImportanceRelax(G, x_good, x, nsamples, influ_obj, if_herd, a)
        
        toc = time.clock()

        print "Iteration: ", iter_num, "    obj = ", obj.item(), "  time = ", (toc - tic),  "   Total/New/Cache: ", influ_obj.itr_total , influ_obj.itr_new , influ_obj.itr_cache

        f.write(str(toc - tic) + " " + str(obj.item()) + " " + str(influ_obj.itr_total) + '/' + str(influ_obj.itr_new) + '/' + str(influ_obj.itr_cache) + "\n") 

        if iter_num % 10 == 0:

            #Round the current solution and get function values
            top_k = Variable(torch.zeros(N)) #conditional grad
            sorted_ind = torch.sort(x, descending = True)[1][0:k]
            top_k[sorted_ind] = 1
            influ = submodObj(G, top_k, p, 100)
            influ_val.append(influ)
            if influ > influ_best:
                influ_best = influ
            influ_val_best.append(influ_best)

    f.close()

    x_opt = x

    #Round the optimum solution and get function values
    top_k = Variable(torch.zeros(N)) #conditional grad
    sorted_ind = torch.sort(x_opt, descending = True)[1][0:k]
    top_k[sorted_ind] = 1
    gt_val = submodObj(G, top_k, p, 100)

    #Save optimum solution and value
    f = open(opt_file, 'w')

    for i in range(len(influ_val)):
        f.write(str(influ_val[i].item()) + ' ' + str(influ_val_best[i].item()) + '\n')

    f.write(str(gt_val.item()) + '\n')
    for x_t in x_opt:
        f.write(str(x_t.item()) + '\n')
    f.close()

    return x_opt

if __name__ == '__main__':

    x = torch.rand(3)
    samples_list = Variable(torch.bernoulli(x.repeat(2, 1)))    
    print getLogProb(samples_list, x)
