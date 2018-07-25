import os
from generate_ground_truth import get_ground_truth
import time
import math
#Hyperparameters

def command_ground_truth():
    #N_list = [6, 7, 8, 9, 10] #in log terms (that is, graph has math.pow(2, N) nodes)
    N_list = [512]
    p = 0.4
    num_fw_iter = 100
    nsamples_mlr_list = [100] 
    num_influ_iter_list = [100]
    k = 20 #cardinality constraint

    command_file = 'jobsQueued.txt'
    f = open(command_file, 'a')
    for N in N_list:
        graph_dir = '/home/pankaj/Sampling/data/input/social_graphs/N_' + str(N) + '/'

        for i in range(10):
            this_file = 'g_N_' + str(N) + '_' + str(i) + '.txt'
            for nsamples_mlr in nsamples_mlr_list:
                for num_influ_iter in num_influ_iter_list:
                    f.write('python generate_ground_truth.py ' + graph_dir + this_file + ' ' + str(k) + ' ' + str(nsamples_mlr) + ' ' + str(num_fw_iter) + ' ' + str(p) + ' ' + str(num_influ_iter) + ' 1\n'  )
    #                print 'python generate_ground_truth.py ' + graph_dir + this_file + ' ' + str(k) + ' ' + str(nsamples_mlr) + ' ' + str(num_fw_iter) + ' ' + str(p) + ' ' + str(num_influ_iter) + ' 0\n'

    f.close()

def command_variance():
    id_list = range(5)
    nsamples_list = [10, 100, 1000, 10000]
    command_file = 'jobsQueued.txt'
    f = open(command_file, 'w')
    for nsamples in nsamples_list:
        for i in id_list:
           print 'python study_diffusion_param.py ' + str(i) + ' ' + str(nsamples)
           f.write('python study_diffusion_param.py ' + str(i) + ' ' + str(nsamples) + '\n')

def study_k_command():
    id_list = range(5)
    k_list = range(20, 300, 50)
    nsamples = 10
    command_file = 'jobsQueued.txt'
    f = open(command_file, 'w')
    for k in k_list:
        for i in id_list:
           print 'python study_diffusion_param.py ' + str(i) + ' ' + str(k)
           f.write('python study_diffusion_param.py ' + str(i) + ' ' + str(k) + '\n')

if __name__ == '__main__':
#    study_k_command()
    command_ground_truth()
