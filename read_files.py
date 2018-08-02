import networkx as nx
import sys
import torch
from torch.autograd import Variable

def read_iterates(filename, N, num_iterates):

    f = open(filename, 'rU')
    x_list = []
    for t in range(num_iterates):
        x = []
        for i in range(N):
            val = float(next(f).strip('\n'))
            x.append(val)
        x_list.append(torch.Tensor(x))
        assert(next(f) == '\n')
    f.close()
    return x_list

def get_sfo_optimum(filename, N):

    x_good = Variable(torch.Tensor([0]*N))

    f = open(filename, 'rU')

    for _ in range(1):
        next(f)

    for line in f:
        num = int(line.strip('\n'))
        x_good[num] = 1

    return x_good

def get_fw_optimum(filename, N):

    x_good = Variable(torch.Tensor([0]*N))

    f = open(filename, 'rU')

    for _ in range(1):
        next(f)

    count = 0

    for line in f:
        x_good[count] = float(line.strip('\n'))
        count += 1

    return x_good

def read_graph(graph_file, N):

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
    
    return G

if __name__ == '__main__':
    N = 512
    g_id =  0
    k = 20
    read_iterates('/home/pankaj/Sampling/data/input/social_graphs/N_' + str(N)
            + '/iterates/g_N_' + str(N) + '_' + str(g_id) + '_' + str(k)
            + '_100_10_0.4_100_0_1_0.txt', N, 2)
