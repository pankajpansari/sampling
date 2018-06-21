import torch
import networkx as nx
import sys
import numpy as np
import torch.optim as optim
import math
import os
import logger
import random
import fnmatch
import time

from graphnet import GraphConv, GraphScorer, MyNet
from torch.autograd import Variable
from influence import ic_model as submodObj
from variance import variance_estimate
from draw_graph import draw
import matplotlib.pyplot as plt
import visdom

random.seed(123)
np.random.seed(123)
graph_dir = '/home/pankaj/Sampling/data/input/social_graphs/N_32/'
torch.set_printoptions(precision = 2)
torch.manual_seed(123)

def getProb(sample, pVec):
    #sample is 0-1 set and pVec is a probability distribution
    temp = pVec * sample + (1 - pVec)*(1 - sample)
    return torch.prod(temp, 1)

def getLogProb(sample, pVec):
    #sample is 0-1 set and pVec is a probability distribution
    temp = pVec * sample + (1 - pVec)*(1 - sample)
    return torch.sum(torch.log(temp), 1)


def kl_loss_mc(input, proposal, graph_file, nsamples):
    #Estimate the objective function using sets from uniform distribution
    batch_size = input.size()[0]
    obj = Variable(torch.FloatTensor([0]*batch_size)) 
    N = int(input.shape[1])

    for t in range(nsamples):
        #draw a sample/set from the uniform distribution
#        sample = Variable(torch.bernoulli(torch.FloatTensor([0.5]*N)))
        sample = Variable(torch.bernoulli(input.squeeze(0).data))
        inputlogP = getLogProb(sample, input)
        proplogP = getLogProb(sample, proposal)
        propP = getProb(sample, proposal)
        inputP = getProb(sample, input)
        print sample, inputP, propP, val
        val = submodObj(graph_file, sample) 
#        obj = torch.add(obj, (propP/uniformP) *(proplogP - (inputlogP + torch.log(val))))
#        print sample.data, inputP.data, propP.data, (propP/inputP).data 
#        print (propP/inputP).data[0] , (proplogP - inputlogP).data[0], torch.log(val).data[0]
        obj = torch.add(obj, (propP/inputP) *(proplogP - (inputlogP + torch.log(val))))
    return obj.mean()/nsamples


def reconstruction_loss(input, proposal):
    #Reconstruction loss - L2 difference between input and proposal 
    batch_size = input.size()[0]
    temp = input - proposal
    l2_norms = torch.norm(temp, 2, 1)

    return ((l2_norms**2).sum())/batch_size

def cross_entropy_loss(input, proposal):
    #Cross-entropy loss between input and proposal 
    batch_size = input.size()[0]
    eps = torch.zeros_like(proposal).fill_(1e-6)    #to avoid numerical issues in log
    term1 = (input * torch.log(proposal + eps)).sum()
    term2 = ((1 - input) * torch.log(1 - proposal + eps)).sum()

    return -(term1 + term2)/batch_size
    
def kl_divergence(input, proposal):
    #kl-divergence between input and proposal 
    batch_size = input.size()[0]
    eps = torch.zeros_like(proposal).fill_(1e-6)    #to avoid numerical issues in log
    term1 = (input * torch.log(proposal + eps)).sum()
    
    for t in range(proposal.shape[0]):
        assert(((1 - proposal[t] + eps[t]) <= 0).sum() == 0) 

    term2 = ((1 - input) * torch.log(1 - proposal + eps)).sum()
    term3 = (input * torch.log(input + eps)).sum()

    assert(torch.isnan(input).sum().item() == 0)
    assert(torch.isnan(proposal).sum().item() == 0)


    assert(torch.isnan(term1).sum().item() == 0)
    assert(torch.isnan(term2).sum().item() == 0)
    assert(torch.isnan(term3).sum().item() == 0)

    return -(term1 + term2 + term3)/batch_size
 
#network_file_list = ['/home/pankaj/Sampling/data/input/social_graphs/k_5/g_k_5_999-network.txt', '/home/pankaj/Sampling/data/input/social_graphs/k_5/g_k_5_998-network.txt']
#batch_size = len(network_file_list) 
def get_data(graph_dir, nNodes, gt_param):

    file_list = os.listdir(graph_dir)
    data = []
    for t in range(1000):
        network_file = 'g_N_32_' + str(t) + '.txt'
        gt_file = 'g_N_32_' + str(t) + '_' + gt_param + '_gt.txt'
        assert(network_file in file_list)
        assert(gt_file in file_list)
        data.append((network_file, gt_file))

    return data

def get_ground_truth(data_list, nNodes):

    ground_truth = Variable(torch.zeros(len(data_list), nNodes))
    for t in range(len(data_list)):
        f = open(graph_dir + data_list[t][1], 'rU')
        count = 0
        for line in f:
            ground_truth[t][count] = float(line)
            count += 1
        assert(torch.isnan(ground_truth[t]).sum().item() == 0)

    f.close()
    return ground_truth

def get_adjacency(data_list, nNodes):

    adjacency = Variable(torch.zeros(len(data_list), nNodes, nNodes))

    for t in range(len(data_list)):
        f = open(graph_dir + data_list[t][0], 'rU')
        adj = np.zeros((nNodes, nNodes))

        for line in f:
            if line.find('Nodes') != -1:
                N = int(line.split(' ')[2])
                assert(N == nNodes)
                break

        for _ in range(1):
            next(f)

        for line in f:
            from_id = int(line.split()[0])
            to_id = int(line.split()[1])
            adj[from_id, to_id] = 1

        f.close()

        adjacency[t] = Variable(torch.from_numpy(adj)).float()

    return adjacency

def get_node_feat(adjacency_list, nNodes):

    num_node_feat = 5
    node_feat = Variable(torch.zeros(len(adjacency_list), nNodes, num_node_feat))

    for t in range(len(adjacency_list)):

        G = nx.DiGraph(adjacency_list[t].numpy())
        
        in_degree = np.array((nx.in_degree_centrality(G)).values())
     
        out_degree = np.array((nx.out_degree_centrality(G)).values())
     
        closeness = np.array((nx.closeness_centrality(G)).values())
     
        between = np.array((nx.betweenness_centrality(G)).values())
     
        pagerank = np.array((nx.pagerank(G)).values())
     
        to_stack = [in_degree, out_degree, closeness, between, pagerank]
     
        assert(len(to_stack) == num_node_feat)
     
        node_feat[t] = Variable(torch.from_numpy(np.stack(to_stack, 1)).float())
 
    return node_feat
       
def round_x(x, N, k):

    assert(x.dim() == 2)
    assert(x.shape[1] == N)

    batch_size = x.shape[0]

    #Round the optimum solution and get function values
    rounded_x = torch.zeros_like(x)
    sorted_ind = torch.sort(x, dim = 1, descending = True)[1][:, 0:k]

    for i in range(batch_size):
        rounded_x[i, sorted_ind[i]] = 1

    return rounded_x

def main():

    lr = float(sys.argv[1])
    mom = float(sys.argv[2])
    n_epochs = int(sys.argv[3])
    total_size = 1000
    train_size = int(sys.argv[4])
    val_size = total_size - train_size
    val_flag = int(sys.argv[5])
   
    k = 9
    net = MyNet(k)

    for params in net.parameters():
        params.requires_grad = True

    loss = kl_divergence
#    loss = reconstruction_loss 
#    optimizer = optim.SGD(net.parameters(), lr=lr, momentum =mom)
#    loss = cross_entropy_loss 
    optimizer = optim.Adam(net.parameters(), lr=lr)
    
    #----------------------------------------------------------
    # Prepare logging
    #----------------------------------------------------------
    env_name = "train_graphnet_1"
    vis = visdom.Visdom(env = env_name)
    vis.text('Graphs of size 32. Using cross-entropy loss without any regularization. Using SGD with given learning rate and momentum.')
    
    # create Experiment
    xp = logger.Experiment(env_name, use_visdom=True,
            visdom_opts={'server': 'http://localhost', 'port': 8097},
            time_indexing=False, xlabel='Epoch')
    
    # log the hyperparameters of the experiment
    xp.log_config({'learning_rate': lr, 'momentum': mom, 'num_epochs': n_epochs, 'train_set_size': train_size, 'val_set_size': total_size - train_size, 'batch_size': train_size})
    
    # create parent metric for training metrics (easier interface)
    #xp.ParentWrapper(tag='train', name='parent', children=(xp.AvgMetric(name='loss')))
    xp.ParentWrapper(tag='train', name='parent', children=(xp.SimpleMetric(name='loss'),))
    xp.ParentWrapper(tag='val', name='parent', children=(xp.SimpleMetric(name='loss'),))
    xp.ParentWrapper(tag='baseline', name='parent', children=(xp.SimpleMetric(name='loss'),))
    
    xp.plotter.set_win_opts(name = 'loss', opts = {'title': 'cross-entropy loss'})
    save_dir = '/home/pankaj/Sampling/data/working/21_06_2018/'
    save_full_name = save_dir + 'training_lr_' + str(lr) + '_mom_' + str(mom) + '_tr_sz_' + str(train_size) + '_' + str(n_epochs)
    gt_param = '9_100_100_0.5_100'
    all_data = get_data(graph_dir, 32, gt_param)
    
    data_size = len(all_data)
    ind_list1 = np.random.choice(range(data_size), size = train_size, replace = False)
    train_data = [all_data[ind] for ind in ind_list1]
    #print train_data
    ind_list2 = [t for t in range(total_size) if t not in ind_list1]
    val_data = [all_data[ind] for ind in ind_list2]

    train_adj = get_adjacency(train_data, 32)
    train_node_feat = get_node_feat(train_adj, 32)

    centrality_train = train_node_feat[:, :, 1]
    factors = 1.0/centrality_train.sum(dim = 1)
    normalised_train = torch.mul(centrality_train, factors.expand_as(centrality_train.t()).t())
    print "Got training data"
    
    
    train_gt_data = get_ground_truth(train_data, 32)
    assert(torch.isnan(train_gt_data).sum().item() == 0)
    
    baseline_train_output = torch.Tensor([float(k)/32]*32).repeat(train_size, 1)
    baseline_val_output = torch.Tensor([float(k)/32]*32).repeat(val_size, 1)

    baseline_loss = loss(train_gt_data, baseline_train_output)
    print "Baseline loss = ", baseline_loss
    
    baseline2_loss = loss(train_gt_data, centrality_train)
#    print normalised_train[2:4]
    print "Baseline2 loss = ", baseline2_loss

    if val_flag == 1:
        val_gt_data = get_ground_truth(val_data, 32)
        val_adj = get_adjacency(val_data, 32)
        val_node_feat = get_node_feat(val_adj, 32)
        centrality_val = val_node_feat[:, :, 1]
        print "Got validation data"
    
    tic = time.time()
    
    bufsize = 0
#    f = open(save_full_name + '_log.txt', 'w', bufsize)

    for epoch in range(0):
#    for epoch in range(n_epochs):
        #get minibatch
        optimizer.zero_grad()   # zero the gradient buffers
    
        train_output = net(train_adj, train_node_feat) 
    
        train_loss = loss(train_gt_data, train_output)
    
        train_loss.backward()
    
#        print net.scorer.t6.grad.data
#        sys.exit()

        optimizer.step()    # Does the update
    
        xp.Parent_Train.update(loss=train_loss)
        xp.Parent_Train.log()
        
        if val_flag == 1:
            val_output = net(val_adj, val_node_feat) 
            val_loss = loss(val_gt_data, val_output)
    
            xp.Parent_Val.update(loss=val_loss)
            xp.Parent_Val.log()
            toc = time.time()
            print "Epoch: ", epoch, "       train loss = ", train_loss.item(), "      val loss = ", val_loss.item()
            f.write(str(epoch) + " " + str(train_loss.item()) + " " + str(val_loss.item()) + " " + str(toc - tic) + '\n')
        else:
            toc = time.time()
            print "Epoch: ", epoch, "       train loss = ", train_loss.item()
            f.write(str(epoch) + " " + str(train_loss.item()) +  " " + str(toc - tic) + '\n')
    
        xp.Parent_Baseline.update(loss=baseline_loss)
        xp.Parent_Baseline.log()
    
#    print "Training done.."
#    torch.save(net, save_full_name + '.net')
#    print "Baseline loss = ", baseline_loss.item()
    
    net = torch.load(save_full_name + '.net')

    #Compare submodular values 
    train_output = net(train_adj, train_node_feat) 

    #Round and query submodular values for training set
    train_output_rounded = round_x(train_output, 32, k)
    train_gt_rounded = round_x(train_gt_data, 32, k)
    baseline_rounded = round_x(baseline_train_output, 32, k)
    baseline2_rounded = round_x(centrality_train, 32, k)

    train_val = torch.zeros(train_size) 
    gt_val = torch.zeros(train_size) 
    baseline_val = torch.zeros(train_size) 
    baseline2_val = torch.zeros(train_size) 

    for i in range(train_size):
        G = nx.DiGraph(train_adj[i].numpy())
        train_val[i] = submodObj(G, train_output_rounded[i], 0.5, 100)
        gt_val[i] = submodObj(G, train_gt_rounded[i], 0.5, 100)
        baseline_val[i] = submodObj(G, baseline_rounded[i], 0.5, 100)
        baseline2_val[i] = submodObj(G, baseline2_rounded[i], 0.5, 100)
    #Query submodular values for ground-truth set

    val_ratio = (train_val/gt_val).mean().item()
    print "Network submodular function ratio (training): ", val_ratio 
    val_ratio = (baseline_val/gt_val).mean().item()
    print "Network submodular function ratio (baseline): ", val_ratio 

    val_ratio = (baseline2_val/gt_val).mean().item()
    print "Network submodular function ratio (baseline 2): ", val_ratio 
    if val_flag == 1:

        val_output = net(val_adj, val_node_feat) 
        #Compare submodular values 
        val_output = net(val_adj, val_node_feat) 
    
        #Round and query submodular values for valing set
        val_output_rounded = round_x(val_output, 32, k)
        val_gt_rounded = round_x(val_gt_data, 32, k)
        baseline_rounded = round_x(baseline_val_output, 32, k)
        baseline2_rounded = round_x(centrality_val, 32, k)
    
        val_val = torch.zeros(val_size) 
        gt_val = torch.zeros(val_size) 
        baseline_val = torch.zeros(val_size) 
        baseline2_val = torch.zeros(val_size) 
    
        for i in range(val_size):
            G = nx.DiGraph(val_adj[i].numpy())
            val_val[i] = submodObj(G, val_output_rounded[i], 0.5, 100)
            gt_val[i] = submodObj(G, val_gt_rounded[i], 0.5, 100)
            baseline_val[i] = submodObj(G, baseline_rounded[i], 0.5, 100)
            baseline2_val[i] = submodObj(G, baseline2_rounded[i], 0.5, 100)
        #Query submodular values for ground-truth set
    
        val_ratio = (val_val/gt_val).mean().item()
        print "Network submodular function ratio (validataion): ", val_ratio 
        val_ratio = (baseline_val/gt_val).mean().item()
        print "Network submodular function ratio (baseline): ", val_ratio 

        val_ratio = (baseline2_val/gt_val).mean().item()
        print "Network submodular function ratio (baseline 2): ", val_ratio 
    sys.exit()
    

   
    print '-'*50
    
    for t in range(10):
        rand_ind = random.randrange(0, 10)
        G = nx.DiGraph(train_adj[t].numpy())
        gt_values = train_gt_data[rand_ind].detach().numpy()
        network_values = train_output[rand_ind].detach().numpy()
        print network_values
    #    if not os.path.exists(save_full_name + '_plots/train'):
    #        os.makedirs(save_full_name + '_plots/train')
    #    graph_filename = train_data[t][0]
    #    draw(G, gt_values, network_values, save_full_name + '_plots/train/' + graph_filename)

    sys.exit()
    for t in range(10):
        rand_ind = random.randrange(0, 10)
        G = nx.DiGraph(val_adj[t].numpy())
        gt_values = val_gt_data[rand_ind].detach().numpy()
        network_values = val_output[rand_ind].detach().numpy()
        if not os.path.exists(save_full_name + '_plots/val'):
            os.makedirs(save_full_name + '_plots/val')
        graph_filename = val_data[t][0]
        draw(G, gt_values, network_values, save_full_name + '_plots/val/' + graph_filename)
 
    xp.to_json(save_full_name + "_val_log.json")
    
    f.close()


if __name__ == '__main__':
    main()
#    test_round()
