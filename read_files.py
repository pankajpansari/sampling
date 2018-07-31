import networkx as nx
import sys

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


