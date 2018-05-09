import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys

class GraphConvLayer(nn.Module):
    def __init__(self, p, w_std, extra_feat_size=0):
        super(GraphConvLayer, self).__init__()
        self.p = p
        self.w_std = w_std

        node_feat = 2 + extra_feat_size
        edge_feat = 2
        self.t1 = nn.Parameter(torch.Tensor(self.p, node_feat))
        self.t2 = nn.Parameter(torch.Tensor(self.p, self.p))
        self.t3 = nn.Parameter(torch.Tensor(self.p, self.p))
        self.t4 = nn.Parameter(torch.Tensor(self.p, edge_feat))

        self.reset()

    def reset(self):
        nn.init.normal(self.t1, mean=0, std=self.w_std)
        nn.init.normal(self.t2, mean=0, std=self.w_std)
        nn.init.normal(self.t3, mean=0, std=self.w_std)
        nn.init.normal(self.t4, mean=0, std=self.w_std)

    def forward(self, node_feat, mu, adjacency, edge_feat):
        batch_size = node_feat.size(0)
        n_node = adjacency.size(1)
        term1 = self.t1.matmul(node_feat)
        term2 = self.t2.matmul(mu).matmul(adjacency)
        term3_1 = F.relu(self.t4.matmul(edge_feat.view(batch_size, 2, n_node * n_node)))
        term3_1 = term3_1.view(batch_size, self.p, n_node, n_node).sum(-1)
        term3 = self.t3.matmul(term3_1)
        
        new_mu = F.relu(term1 + term2 + term3)

        return new_mu

class GraphConv(nn.Module):
    def __init__(self, n_layer, p, w_std, extra_feat_size=0):
        super(GraphConv, self).__init__()
        self.p = p
        self.extra_feat_size = extra_feat_size

        self.layers = nn.ModuleList()
        for i in range(n_layer):
            self.layers.add_module(str(i), GraphConvLayer(p, w_std, extra_feat_size))

    def forward(self, x, adjacency, weights, extra_feat):
        assert(x.ndimension() == 2)
        assert(adjacency.ndimension() == 3)
        assert(weights.ndimension() == 3)
        batch_size = x.size(0)
        n_node = adjacency.size(1)
        # Node features
        to_stack = []
        to_stack.append(x)
        to_stack.append(Variable(x.data.new([1]).expand_as(x)))
#        print to_stack
        if extra_feat is not None:
            if extra_feat.ndimension() == 2:
                assert(self.extra_feat_size == 1)
                to_stack.append(extra_feat)
            else:
                assert(extra_feat.ndimension() == 3)
                assert(extra_feat.size(2) == self.extra_feat_size)
                for i in range(extra_feat.size(2)):
                    to_stack.append(extra_feat.select(2, i))

        node_feat = torch.stack(to_stack, 1).float()

        # Edge features
#        temp = x.unsqueeze(1)
#        temp = temp.repeat(1, n_node, 1) 
#        to_stack_edge = []
#        for i in range(temp.size(0)):
#            to_stack_edge.append(temp.select(0, i).t() - temp.select(0, i))
#        x_diff_feat =  torch.stack(to_stack_edge, 0)

        edge_feat = x.data.new(batch_size, 2, n_node, n_node)
        edge_feat.select(1, 0).copy_(weights.data)
        edge_feat.select(1, 1).fill_(1)
#        edge_feat.select(3, 2).copy_(x_diff_feat.data)
        edge_feat = Variable(edge_feat)

        mu = Variable(x.data.new(batch_size, self.p, n_node).zero_())

        for layer in self.layers:
            mu = layer(node_feat, mu, adjacency, edge_feat)

        return mu

class GraphScorer(nn.Module):
    def __init__(self, p, w_std):
        super(GraphScorer, self).__init__()
        self.p = p
        self.w_std = w_std

        self.t5_1 = nn.Parameter(torch.Tensor(1, self.p))
        self.t5_2 = nn.Parameter(torch.Tensor(1, self.p))
        self.t6 = nn.Parameter(torch.Tensor(self.p, self.p))
        self.t7 = nn.Parameter(torch.Tensor(self.p, self.p))

        self.reset()

    def reset(self):
        nn.init.normal(self.t5_1, mean=0, std=self.w_std)
        nn.init.normal(self.t5_2, mean=0, std=self.w_std)
        nn.init.normal(self.t6, mean=0, std=self.w_std)
        nn.init.normal(self.t7, mean=0, std=self.w_std)

    def forward(self, mu):
        accum = mu.sum(-1, keepdim=True)

        term1 = F.relu(self.t6.matmul(accum))
        term2 = F.relu(self.t7.matmul(mu))

        g_score = self.t5_1.matmul(term1).squeeze(1)
        per_node_score = self.t5_2.matmul(term2).squeeze(1)

        output = g_score.expand_as(per_node_score) + per_node_score

        output_distribution = torch.sigmoid(output)

        return output_distribution

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        
        n_layer = 3
        p = 32
        w_scale = 1e-3  
        extra_feat = 6
        self.conv = GraphConv(n_layer, p, w_scale, extra_feat)
        self.scorer = GraphScorer(p, w_scale)

    def forward(self, S, adjacency, weights, extra, choice=None):
#        adjacency_v = Variable(adjacency)
#        weights_v = Variable(weights)
#        mu = self.conv(S, adjacency_v, weights_v, extra)
        mu = self.conv(S, adjacency, weights, extra)
        scores = self.scorer(mu)
        return scores
#        if choice is not None:
#            topk_val = scores.gather(1, choice)
#            topk_ind = choice
#        else:
#            topk_val, topk_ind = torch.topk(scores, 3, dim=1)
#            topk_ind, _ = torch.sort(topk_ind.squeeze_(0))
#
#        return topk_val.sum(-1), topk_ind
