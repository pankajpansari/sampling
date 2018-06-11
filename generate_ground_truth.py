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

    G = nx.DiGraph()

    for line in f:
        if line.find('Nodes') != -1:
            N = int(line.split(' ')[2])
            G.add_nodes_from(range(N))
            break

    for _ in range(1):
        next(f)

    for line in f:
        from_id = int(line.split()[0])
        to_id = int(line.split()[1])
        G.add_edge(from_id, to_id)

  
    ind = filename.find('-')
    file_prefix = filename[0:ind] + '_' + str(k) + '_' + str(nsamples_mlr) + '_' + str(num_fw_iter) + '_' + str(p) + '_' + str(num_influ_iter) 
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
