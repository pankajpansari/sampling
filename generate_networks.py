from subprocess import call
import os
import math
import sys
import numpy as np

seed_m = np.array([[0.994, 0.384], [0.414, 0.249]])
seed_text = '"' + str(seed_m[0, 0]) + ' ' + str(seed_m[0, 1]) + '; ' + str(seed_m[1, 0]) + ' ' + str(seed_m[1, 1]) + '"'
seed_value = 123
num_graphs = 10 
logN_list = [9]

FNULL = open(os.devnull, 'w')
for logN in logN_list: 
    n_nodes = int(math.pow(2, logN))
    n_edges = int(math.pow(np.sum(seed_m), logN))
    destination_dir = '/home/pankaj/Sampling/data/input/social_graphs/answers_synthetic/N_' + str(n_nodes) + '/'
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    print "Generating " + str(num_graphs) + " kronecker graphs with #nodes: " + str(n_nodes) + " #edges: " + str(n_edges)
    for t in range(num_graphs):
        if t % 100 == 0:
            print ".", 
        filename = destination_dir + "g_N_" + str(n_nodes) + '_' + str(t) + '.txt'
        seed_value = seed_value + 1
        command = '/home/pankaj/Sampling/code/fw_social_networks/snap/examples/krongen/krongen -m:' + seed_text + ' -i:' + str(logN) + ' -o:' + filename + ' -s:' + str(seed_value)
#        print command
        call(command, stdout=FNULL, shell=True)
    print "Done"

