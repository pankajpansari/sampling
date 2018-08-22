import os
import time
import math
import random
import networkx as nx
import matplotlib.pyplot as plt
from read_files import *

def degree_facebook():
    graph_file = '/home/pankaj/Sampling/data/input/social_graphs/facebook/facebook_combined.txt'
    N = 4039

    G = read_facebook_graph(graph_file, N)

    degree_dict = nx.out_degree_centrality(G)
    
    degree_values = [N*x for x in degree_dict.values()]
    
#    print degree_dict.keys()[0:10], degree_dict.values()[0:10]


    plt.hist(degree_values, bins = range(0, 350, 10))
    plt.show()

def degree_synthetic():
    graph_dir = '/home/pankaj/Sampling/data/input/social_graphs/' 

    N_list = [32, 64, 128, 256]
    for N in N_list:
        for t in range(16):

            p = random.choice(range(1000))

            f = open(graph_dir + 'N_' + str(N) + '/g_N_' + str(N) + '_' + str(p) + '.txt', 'rU')

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

            f.close()

            N = nx.number_of_nodes(G)

            degree_tuple = nx.degree(G)


            max_degree = 0
            for item in degree_tuple:
                if item[1] > max_degree:
                    max_degree = item[1]

            degree_count = [0]*(max_degree + 1) #+1 to include degree 0

            for item in degree_tuple:
                degree_count[item[1]] += 1

            plt.subplot(4, 4, t + 1)
            plt.plot(degree_count)

        plt.savefig('/home/pankaj/Sampling/data/working/28_06_2018/degree_plot_' + str(N) + '.jpg') 
        plt.clf()

if __name__ == '__main__':
    degree_facebook()
