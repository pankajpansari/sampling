import networkx as nx
import ndlib.models.epidemics.SIRModel as sir
import ndlib.models.epidemics.IndependentCascadesModel as ids
import numpy as np
import sys
import ndlib.models.ModelConfig as mc
import time
import random
import torch
from torch.autograd import Variable

random.seed(1234)

class Influence(object):
    def __init__(self, G, p, niter):
        self.cache = {}
        self.cache_hits = 0
        self.G = G
        self.niter = niter 
        self.p = p 

    def reset(self):
        self.cache.reset()
        self.cache_hits = 0

    def __call__(self, sample):

#        one_hot = sample.byte() 
        key = sample.tobytes()

#        val = ic_model(self.G, sample)
#        self.cache[key] = val
        if key not in self.cache:
            val = ic_model(self.G, sample, self.p, self.niter)
            self.cache[key] = val
        else:
            self.cache_hits += 1

        return self.cache[key]

###########################################
def ic_model(G, sample, p, iterations):
#Independent Cascade Model

    N = G.number_of_nodes()

    seed_set = [i for i in range(N) if sample[i] == True]
    avg_influence = Variable(torch.FloatTensor([0]), requires_grad = True) 

    for j in range(iterations):            
        S=list(seed_set)
        for i in range(len(S)):                 # Process each node in seed set
            for neighbor in G.neighbors(S[i]):    
                if random.random()<p:           # Generate a random number and compare it with propagation probability
                    if neighbor not in S:       
                        S.append(neighbor)
        avg_influence = avg_influence + (float(len(S))/iterations) 
#    print 'Total influence:',int(round(avg_influence.item()))
    return avg_influence

def main():
    g = nx.erdos_renyi_graph(1000, 0.2)
    tic = time.clock()
    ic_model(g)
    toc = time.clock()
    print "Time elapsed = ", toc - tic

if __name__ == '__main__':
    main()
