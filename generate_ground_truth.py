import networkx as nx
import sys
import numpy as np
import math
import torch
from influence import ic_model as submodObj
from torch.autograd import Variable
from frank_wolfe import runFrankWolfe
from frank_wolfe_importance import runImportanceFrankWolfe
from read_files import get_sfo_optimum, get_fw_optimum, read_graph
import time
import argparse
np.random.seed(1234)
torch.manual_seed(1234) 

def get_variance(N, g_id, k, nsamples, num_fw_iter, p, num_influ_iter, if_herd):

    graph_file = '/home/pankaj/Sampling/data/input/social_graphs/N_' + str(N) + '/graphs/g_N_' + str(N) + '_' + str(g_id) + '.txt'

    G = read_graph(graph_file, N)

    x_good_sfo = get_sfo_optimum('/home/pankaj/Sampling/data/input/social_graphs/N_' + str(N) + 'sfo_gt/g_N_' + str(N) + '_id_' + str(g_id) + '_k_' + str(k) + '.txt', N) 

    x_good_fw = get_fw_optimum('/home/pankaj/Sampling/data/input/social_graphs/N_' + str(N) + 'fw_gt/g_N_' + str(N) + '_' + str(g_id) + '_' + str(k) + '.txt', N) 

    temp = '/home/pankaj/Sampling/data/input/social_graphs/N_' + str(N) + '/var_study/g_N_' + str(N) + '_' + str(g_id) 

    var_file = '_'.join(str(x) for x in [temp, k, nsamples_mlr, num_fw_iter, p, num_influ_iter, if_herd]) + '.txt'

    variance_study(G, nsamples, k, var_file, num_fw_iter, p, num_influ_iter, if_herd, x_good_sfo, x_good_fw) 


def get_ground_truth(N, g_id, k, nsamples_mlr, num_fw_iter, p, num_influ_iter, if_herd, if_importance, if_sfo_gt):

    graph_file = '/home/pankaj/Sampling/data/input/social_graphs/N_' + str(N) + '/graphs/g_N_' + str(N) + '_' + str(g_id) + '.txt'

    G = read_graph(graph_file, N)

    temp = '/home/pankaj/Sampling/data/input/social_graphs/N_' + str(N) + '/fw_log/g_N_' + str(N) + '_' + str(g_id) 

    log_file = '_'.join(str(x) for x in [temp, k, nsamples_mlr, num_fw_iter, p, num_influ_iter, if_herd, if_importance, if_sfo_gt]) + '.txt'

    if if_importance == 1:

        if if_sfo_gt == 1:

            x_good = get_sfo_optimum('/home/pankaj/Sampling/data/input/social_graphs/N_' + str(N) + 'sfo_gt/g_N_' + str(N) + '_id_' + str(g_id) + '_k_' + str(k) + '.txt', N) 

        else:
            x_good = get_fw_optimum('/home/pankaj/Sampling/data/input/social_graphs/N_' + str(N) + 'fw_gt/g_N_' + str(N) + '_id_' + str(g_id) + '_k_' + str(k) + '_100.txt', N) 

        x_opt = runImportanceFrankWolfe(G, nsamples_mlr, k, log_file, num_fw_iter, p, num_influ_iter, if_herd, x_good)

    else:
        x_opt = runFrankWolfe(G, nsamples_mlr, k, log_file, num_fw_iter, p, num_influ_iter, if_herd)

    #Round the optimum solution and get function values
    top_k = Variable(torch.zeros(N)) #conditional grad
    sorted_ind = torch.sort(x_opt, descending = True)[1][0:k]
    top_k[sorted_ind] = 1
    gt_val = submodObj(G, top_k, p, num_influ_iter)

    #Save optimum solution and value
    temp = '/home/pankaj/Sampling/data/input/social_graphs/N_' + str(N) + '/fw_opt/g_N_' + str(N) + '_' + str(g_id) 

    opt_file = '_'.join(str(x) for x in [temp, k, nsamples_mlr, num_fw_iter, p,
        num_influ_iter, if_herd, if_importance, if_sfo_gt]) + '.txt'

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
    parser.add_argument('if_sfo_gt', help='True if greedy ground-truth to be used during importance sampling', type=int)

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
    if_sfo_gt = args.if_sfo_gt

#    get_variance(N, g_id, k, nsamples_mlr, num_fw_iter, p, num_influ_iter, if_herd)
    get_ground_truth(N, g_id, k, nsamples_mlr, num_fw_iter, p, num_influ_iter, if_herd, if_importance, if_sfo_gt)

    print "Compeleted in " + str(time.clock() - tic) + 's'

if __name__ == '__main__':
    main()
