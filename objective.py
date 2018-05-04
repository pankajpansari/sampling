from subprocess import call
import os
import sys
import torch
from torch.autograd import Variable

def submodObj(graph_file, sample):
    N = len(sample.data)
    this_set = [str(i) for i in range(N) if sample.data[i] == 1] 
    subset = ';'.join(this_set)
    subset = '"' + subset + '"'
    subset_len = len(this_set)
    influmax_command = './influmax/./influmax -n:' + graph_file +  ' -s:' + str(subset_len) + ' -b:4 -ls:' + str(subset)
    FNULL = open(os.devnull, 'w')
    call(influmax_command, stdout=FNULL, shell=True)
    f = open('./influmax/influence-info-network.txt', 'rU')
    count = 0
    submodular_val = -1
    for line in f:
        count += 1;
        if count == 2:
            line.strip('\n')
            submodular_val = float(line)
            break
    
    return submodular_val 

def main():
    N = 10 
    sample = Variable(torch.bernoulli(torch.FloatTensor([0.5]*N)))
    graph_file = './influmax/example-network.txt'
    print submodObj(graph_file, sample)

if __name__ == '__main__':
    main()
