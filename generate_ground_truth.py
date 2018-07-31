import networkx as nx
import sys
import numpy as np
import math
import torch
from influence import ic_model as submodObj
from torch.autograd import Variable
from frank_wolfe import runFrankWolfe
from frank_wolfe_importance import runImportanceFrankWolfe
import time
import argparse
np.random.seed(1234)
torch.manual_seed(1234) 

def get_ground_truth(N, g_id, k, nsamples_mlr, num_fw_iter, p, num_influ_iter, if_herd, if_importance):

    graph_file = '/home/pankaj/Sampling/data/input/social_graphs/N_' + str(N) + '/graphs/g_N_' + str(N) + '_' + str(g_id) + '.txt'

    f = open(graph_file, 'rU')

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

    N = nx.number_of_nodes(G)

    temp = '/home/pankaj/Sampling/data/input/social_graphs/N_' + str(N) + '/fw_log/g_N_' + str(N) + '_' + str(g_id) 

    log_file = '_'.join(str(x) for x in [temp, k, nsamples_mlr, num_fw_iter, p, num_influ_iter, if_herd, if_importance]) + '.txt'

    if if_importance == 1:
        x_opt = runImportanceFrankWolfe(G, nsamples_mlr, k, log_file, num_fw_iter, p, num_influ_iter, if_herd)
    else:
        x_opt = runFrankWolfe(G, nsamples_mlr, k, log_file, num_fw_iter, p, num_influ_iter, if_herd)

    #Round the optimum solution and get function values
    top_k = Variable(torch.zeros(N)) #conditional grad
    sorted_ind = torch.sort(x_opt, descending = True)[1][0:k]
    top_k[sorted_ind] = 1
    gt_val = submodObj(G, top_k, p, num_influ_iter)

    #Save optimum solution and value
    temp = '/home/pankaj/Sampling/data/input/social_graphs/N_' + str(N) + '/fw_gt/g_N_' + str(N) + '_' + str(g_id) 

    opt_file = '_'.join(str(x) for x in [temp, k, nsamples_mlr, num_fw_iter, p, num_influ_iter, if_herd, if_importance]) + '.txt'

    f = open(opt_file, 'w')
    f.write(str(gt_val.item()) + '\n')
    for x_t in x_opt:
        f.write(str(x_t.item()) + '\n')
    f.close()

def main():

    tic = time.clock()
    parser = argparse.ArgumentParser(description='Generating ground truth for given graph using Frank-Wolfe')
    parser.add_argument('N', help='Number of nodes in graph', type=int)
    parser.add_argument('g_id', help='Id of the graph file', type=int)
    parser.add_argument('k', help='Cardinality constraint', type=int)
    parser.add_argument('nsamples_mlr', help='Number of samples for multilinear relaxation estimation', type=int)
    parser.add_argument('num_fw_iter', help='Number of iterations of Frank-Wolfe', type=int)
    parser.add_argument('p', help='Propagation probability for diffusion model', type=float)
    parser.add_argument('num_influ_iter', help='Number of iterations of independent-cascade diffusion', type=int)
    parser.add_argument('if_herd', help='True if herding', type=int)
    parser.add_argument('if_importance', help='True if importance sampling to be done', type=int)

    args = parser.parse_args()
    
    N = args.N
    g_id = args.g_id
    k = args.k #cardinality constraint
    nsamples_mlr = args.nsamples_mlr #draw these many sets from x for multilinear relaxation
    num_fw_iter = args.num_fw_iter 
    p = args.p 
    num_influ_iter = args.num_influ_iter 
    if_herd = args.if_herd
    if_importance = args.if_importance

    get_ground_truth(N, g_id, k, nsamples_mlr, num_fw_iter, p, num_influ_iter, if_herd, if_importance)

    print "Compeleted in " + str(time.clock() - tic) + 's'

if __name__ == '__main__':
    main()
