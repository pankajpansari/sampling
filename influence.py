import networkx as nx
import ndlib.models.epidemics.SIRModel as sir
import ndlib.models.epidemics.IndependentCascadesModel as ids
import numpy as np
import sys
import ndlib.models.ModelConfig as mc
import time
import random

np.random.seed(1)
 
def ic_model(G):
#Independent Cascade Model

    iterations=1000                       # Number of Iterations
    p=.01                                  # Propagation probability
    seed_set=random.sample(G.nodes(),10)    # Selecting intial seed set randomly
    print 'Selected Seeds:',seed_set
    avg_influence=0.0
    for i in range(iterations):            
        S=list(seed_set)
        for i in range(len(S)):                 # Process each node in seed set
            for neighbor in G.neighbors(S[i]):    
                if random.random()<p:           # Generate a random number and compare it with propagation probability
                    if neighbor not in S:       
                        S.append(neighbor)
        avg_influence+=(float(len(S))/iterations) 
    print 'Total influence:',int(round(avg_influence))

def main():
    g = nx.erdos_renyi_graph(1000, 0.2)
    tic = time.clock()
    ic_model(g)
    toc = time.clock()
    print "Time elapsed = ", toc - tic

if __name__ == '__main__':
    main()
