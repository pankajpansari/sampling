from __future__ import print_function
import os
from generate_ground_truth import get_ground_truth
import time
import math
#Hyperparameters

def command_ground_truth():
    #N_list = [6, 7, 8, 9, 10] #in log terms (that is, graph has math.pow(2, N) nodes)
    N = 512
    p = 0.4
    num_fw_iter = 10 
    nsamples_mlr_list = [5, 10, 20] 
    num_influ_iter = 100
    k = 20 #cardinality constraint
    a_list = [1e-1, 1e-2, 1e-3] 

    command_file = 'jobsQueued.txt'
    f = open(command_file, 'a')
    for i in range(3):
        for nsamples in nsamples_mlr_list:
            #simple mc
            print('python generate_ground_truth.py ' + ' '.join(str(x) for x in
                [N, i, k, nsamples, num_fw_iter, p, num_influ_iter, 0, 0, 0,
                    1.0]), file = f)

#            for a in a_list:
#                #no herding
#                print('python generate_ground_truth.py ' + ' '.join(str(x) for
#                    x in [N, i, k, nsamples, num_fw_iter, p, num_influ_iter, 0,
#                        1, 1, a]), file = f)
#
#                print('python generate_ground_truth.py ' + ' '.join(str(x) for
#                    x in [N, i, k, nsamples, num_fw_iter, p, num_influ_iter, 0,
#                        1, 0, a]), file = f)

#                #herding
#                print 'python generate_ground_truth.py ' + ' '.join(str(x) for x in [N, i, k, nsamples, num_fw_iter, p, num_influ_iter, 1, 1, 1, a])
#
#                print 'python generate_ground_truth.py ' + ' '.join(str(x) for x in [N, i, k, nsamples, num_fw_iter, p, num_influ_iter, 1, 1, 0, a])
# 

    f.close()

#def command_variance():
#    id_list = range(5)
#    nsamples_list = [10, 100, 1000, 10000]
#    command_file = 'jobsQueued.txt'
#    f = open(command_file, 'w')
#    for nsamples in nsamples_list:
#        for i in id_list:
#           print 'python study_diffusion_param.py ' + str(i) + ' ' + str(nsamples)
#           f.write('python study_diffusion_param.py ' + str(i) + ' ' + str(nsamples) + '\n')
#
#def study_k_command():
#    id_list = range(5)
#    k_list = range(20, 300, 50)
#    nsamples = 10
#    command_file = 'jobsQueued.txt'
#    f = open(command_file, 'w')
#    for k in k_list:
#        for i in id_list:
#           print 'python study_diffusion_param.py ' + str(i) + ' ' + str(k)
#           f.write('python study_diffusion_param.py ' + str(i) + ' ' + str(k) + '\n')

if __name__ == '__main__':
#    study_k_command()
    command_ground_truth()
