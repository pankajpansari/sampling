import networkx as nx
import sys
import numpy as np
import math
import torch
from influence import ic_model as submodObj
from torch.autograd import Variable
from frank_wolfe import runFrankWolfe, getGrad
import time
import argparse
np.random.seed(1234)
torch.manual_seed(1234) 


def get_ground_truth(filename, k, nsamples_mlr, num_fw_iter, p, num_influ_iter):
    f = open(filename, 'rU')
    nNodes = 0
    for line in f:
        if line == '\n':
            break
        else:
            nNodes += 1
    
    alphas = np.zeros((nNodes, nNodes))
    
    for line in f:
        temp = line.strip('\n')
        temp2 = temp.split(",")
        a = int(temp2[0])
        b = int(temp2[1])
        alphas[a, b] = float(temp2[2])
    
    max_alpha = np.max(alphas)
    #alphas are affinity measures - take -ve to get a dissimilarity measure
    weights = max_alpha - alphas
    w = np.asmatrix(weights)
    G = nx.DiGraph(w)
    
    N = nx.number_of_nodes(G)
    
    ind = filename.find('-')
    file_prefix = filename[0:ind]
    
    runFrankWolfe(G, nsamples_mlr, k, file_prefix, num_fw_iter, p, num_influ_iter)

def main():

    tic = time.clock()
    parser = argparse.ArgumentParser(description='Generating ground truth for given graph using Frank-Wolfe')
    parser.add_argument('filename', help='Full path the graph file', type=str)
    parser.add_argument('k', help='Cardinality constraint', type=int)
    parser.add_argument('nsamples_mlr', help='Number of samples for multilinear relaxation estimation', type=int)
    parser.add_argument('num_fw_iter', help='Number of iterations of Frank-Wolfe', type=int)
    parser.add_argument('p', help='Propagation probability for diffusion model', type=float)
    parser.add_argument('num_influ_iter', help='Number of iterations of independent-cascade diffusion', type=int)
    args = parser.parse_args()
    
    filename = args.filename
    k = args.k #cardinality constraint
    nsamples_mlr = args.nsamples_mlr #draw these many sets from x for multilinear relaxation
    num_fw_iter = args.num_fw_iter 
    p = args.p 
    num_influ_iter = args.num_influ_iter 

    get_ground_truth(filename, k, nsamples_mlr, num_fw_iter, p, num_influ_iter)
    print filename + " compeleted in " + str(time.clock() - tic) + 's'
if __name__ == '__main__':
    main()
