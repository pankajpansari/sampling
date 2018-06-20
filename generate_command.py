import os
from generate_ground_truth import get_ground_truth
import time
import math
#Hyperparameters

#N_list = [6, 7, 8, 9, 10] #in log terms (that is, graph has math.pow(2, N) nodes)
N_list = [32]
p = 0.5
num_fw_iter = 100
#nsamples_mlr_list = [10, 100, 1000] #draw these many sets from x for multilinear relaxation
nsamples_mlr_list = [100] 
num_influ_iter_list = [100]

#graph_dir = '/home/pankaj/Sampling/data/input/social_graphs/k_' + str(N) + '/'
command_file = 'jobsQueued.txt'
f = open(command_file, 'w')
for N in N_list:
    graph_dir = '/home/pankaj/Sampling/data/input/social_graphs/N_' + str(N) + '/'
    file_list = os.listdir(graph_dir)

    k = 9 #cardinality constraint

    for this_file in file_list:
        if 'log' not in this_file and 'gt' not in this_file:
            for nsamples_mlr in nsamples_mlr_list:
                for num_influ_iter in num_influ_iter_list:
                    f.write('python generate_ground_truth.py ' + graph_dir + this_file + ' ' + str(k) + ' ' + str(nsamples_mlr) + ' ' + str(num_fw_iter) + ' ' + str(p) + ' ' + str(num_influ_iter) + '\n'  )
#                    print 'python generate_ground_truth.py ' + graph_dir + this_file + ' ' + str(k) + ' ' + str(nsamples_mlr) + ' ' + str(num_fw_iter) + ' ' + str(p) + ' ' + str(num_influ_iter) + '\n'

f.close()
