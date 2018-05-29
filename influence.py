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

np.random.seed(1)
 
def ic_model(G, sample, p = 0.01):
#Independent Cascade Model

    N = G.number_of_nodes()
    iterations = 100                       # Number of Iterations

    seed_set = [i for i in range(N) if sample.data[i] == 1]

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
