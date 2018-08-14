#!/usr/bin/env python

from __future__ import division
# import modules used here -- sys is a very standard one
import sys
import os
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats 

def plotHistogram(a, start, end):

    x = a.flatten()
    print x.shape
    minVal = x.min()
    maxVal = x.max()
    print minVal
    print maxVal
    modeVal = stats.mode(x)[0]
    print modeVal 
    t = (end*minVal - start*maxVal)/(end - start)
    p = (maxVal - minVal)/(end - start)
    y = (x - t)/p
    transformy = np.tan((y + 0.5)*math.pi)
    transformy = np.log(y+start)
    plt.hist(transformy, 1000, log=True)

    plt.title("Negative Gradient Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
#    plt.savefig('gradient_histogram_transform.pdf')
    plt.show()
    
def plotHeatmap(a):
    print a.shape
    category = 1 
#    for category in range(21):
    b = a[category, :]
    b = b.reshape(320, 213) 
    print b.shape
    plt.imshow(b, cmap = 'hot', interpolation = 'nearest')
    plt.colorbar()
    plt.savefig('negGrad_heatmap_grass.pdf')
#    plt.show()

def linePlot(y, z, filename):
     plt.plot(y, linestyle='-', color = 'r', label = 'training') 
     plt.plot(z, linestyle='-', color = 'b', label = 'validation') 
     plt.xlabel('Iterations')
     plt.ylabel('KL-divergence')
     plt.legend()
     plt.show()

def linePlotShow(x):
     plt.plot(x, linestyle='-') 
     plt.xlabel('Iterations')
     plt.ylabel('Obj function val')
#     plt.ylim([0, 100])
#     plt.axhline(y[0], linewidth=2, color = 'b')
#     plt.xlim([-1, 1])
     plt.show()

def linePlotSave(x, filename):
     plt.plot(x, linestyle='-') 
     plt.xlabel('Iterations')
     plt.ylabel('Obj function val')
#     plt.ylim([0, 100])
#     plt.axhline(y[0], linewidth=2, color = 'b')
#     plt.xlim([-1, 1])
     plt.savefig(filename)

def difference(a):
     b = a[a[:, 0] == 0].shape[0]
     diff = np.zeros(b)
     ngraphs = 2
     for i in range(ngraphs):
         ind = a[:, 0] == i 
         diff += a[ind, 2] - a[ind, 3]
     diff = diff/ngraphs
     for i in range(len(diff)):
         print i+1, diff[i]
     plt.plot(diff)
     plt.xlabel('Cardinality Constraint')
     plt.ylabel('Difference')
     plt.title('Plot of difference in submodular function values between optimum and best of 20 randomly drawn sets (averaged over 8 graphs of 32 nodes)')
     plt.savefig(sys.argv[2])

def plot_iterates_hist():
    filename = '/home/pankaj/Sampling/data/input/social_graphs/N_512/iterates/' + sys.argv[1]
    a = np.loadtxt(filename)
    for i in range(10):
        plt.subplot(5, 2, i + 1)
        temp = a[512*i:512*(i + 1)]
        plt.hist(temp, bins = 100, range = (0, 1))
        plt.title("iter = " + str(i))
    plt.show()
#    plt.savefig(sys.argv[2])

def plot_solution_variance():

    solution_dir = '/home/pankaj/Sampling/data/input/social_graphs/N_512/fw_opt/'
    file_list = os.listdir(solution_dir)
    key_list = ['0_0_0.0', '0_0_0.1', '0_1_0.1']
    label_list = ['simple MC', 'convex with fw opt. 0.1', 'convex with greedy opt. 0.1']
    for p in range(3):
        d = []
        e = []
        nsamples_list = [1, 5, 10, 20]
        for nsamples in nsamples_list:
            a = []
            c = []
            for g_id in range(5):
                b = []
                for t in range(5):
                    seed = 123 + t 
                    opt_file = 'g_N_512_' + str(g_id) + '_20_' + str(nsamples) + '_20_0.4_100_' + str(key_list[p]) + '_' + str(seed) + '.txt'
                    if opt_file not in file_list:
                        opt_file = 'g_N_512_' + str(g_id) + '_20_' + str(nsamples) + '_10_0.4_100_' + str(key_list[p]) + '_' + str(seed) + '.txt'
                        if opt_file not in file_list:
                            print opt_file
                            sys.exit()

                    f = open(solution_dir + opt_file, 'r')
                    opt = float(next(f))
                    f.close()
                    b.append(opt)
                a.append(np.var(b))
                c.append(np.mean(b))
            d.append(np.mean(a))
            e.append(np.mean(c))
        print d, e
        plt.subplot(2, 1, 1)
        plt.plot(nsamples_list, d, label = label_list[p])
        plt.xlabel('# samples for relaxation estimation')
        plt.ylabel('Variance of 5 rounded solutions (averaged over 5 graphs)')
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(nsamples_list, e, label = label_list[p])
        plt.xlabel('# samples for relaxation estimation')
        plt.ylabel('Mean of 5 rounded solutions (averaged over 5 graphs)')
        plt.legend()
#    plt.savefig('variance_solution.jpg')
    plt.show()

def plot_email_solution_variance():

    solution_dir = '/home/pankaj/Sampling/data/working/13-08-2018-email_sol_var_mc/workspace/'
    file_list = os.listdir(solution_dir)
    key_list = ['0_0_0.0']
    label_list = ['simple MC']
    for p in range(1):
        d = []
        e = []
        nsamples_list = [1, 5]
        for nsamples in nsamples_list:
            b = []
            for t in range(5):
                seed = 123 + t 
                opt_file = 'fw_opt_20_' + str(nsamples) + '_10_0.4_200_' + str(key_list[p]) + '_' + str(seed) + '.txt'
                if opt_file not in file_list:
                    print opt_file
                    sys.exit()

                f = open(solution_dir + opt_file, 'r')
                opt = float(next(f))
                f.close()
                b.append(opt)
            d.append(np.var(b))
            e.append(np.mean(b))
        print d, e
        plt.subplot(2, 1, 1)
        plt.plot(nsamples_list, d, label = label_list[p], marker = '+', markersize = 10)
        plt.xlabel('# samples for relaxation estimation')
        plt.ylabel('Variance of 5 rounded solutions (1 graph, 10 FW iter)')
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(nsamples_list, e, label = label_list[p], marker = '+', markersize = 10)
        plt.xlabel('# samples for relaxation estimation')
        plt.ylabel('Variance of 5 rounded solutions (1 graph, 10 FW iter)')
        plt.legend()
#    plt.savefig('variance_solution.jpg')
    plt.show()

def parse_email_timing():

    solution_dir = '/home/pankaj/Sampling/data/working/13-08-2018-email_sol_var_mc/workspace/'
    file_list = os.listdir(solution_dir)
    key_list = ['0_0_0.0']
    label_list = ['simple MC']
    for p in range(1):
        d = []
        nsamples_list = [1, 5]
        for nsamples in nsamples_list:
            b = []
            for t in range(5):
                seed = 123 + t 
                opt_file = 'fw_log_20_' + str(nsamples) + '_10_0.4_200_' + str(key_list[p]) + '_' + str(seed) + '.txt'
                if opt_file not in file_list:
                    print opt_file
                    sys.exit()

                f = open(solution_dir + opt_file, 'r')
                for i in range(9):
                    next(f)
                timing = float(next(f).split(' ')[0])
                f.close()
                print nsamples, timing
                b.append(timing)
            d.append(np.mean(b))
        print d

def plot_email_obj():

    solution_dir = '/home/pankaj/Sampling/data/working/13-08-2018-email_sol_var_mc/workspace/'
    file_list = os.listdir(solution_dir)
    key_list = ['0_0_0.0']
    label_list = ['simple MC']
    for p in range(1):
        nsamples_list = [5]
        for nsamples in nsamples_list:
            for t in range(5):
                b = []
                seed = 123 + t 
                opt_file = 'fw_log_20_' + str(nsamples) + '_10_0.4_200_' + str(key_list[p]) + '_' + str(seed) + '.txt'
                if opt_file not in file_list:
                    print opt_file
                    sys.exit()

                f = open(solution_dir + opt_file, 'r')
                for i in range(10):
                    obj = float(next(f).split(' ')[1])
                    b.append(obj)
                f.close()
                plt.plot(range(1, 11), b, linestyle = '-')
    plt.xlabel('FW iterations')
    plt.ylabel('Relaxation objective')
    plt.show()

def plot_facebook_obj():

    solution_dir = '/home/pankaj/Sampling/data/working/13-08-2018-facebook_sol_var_mc/workspace/'
    file_list = os.listdir(solution_dir)
    key_list = ['0_0_0.0']
    label_list = ['simple MC']
    for p in range(1):
        nsamples_list = [1]
        for nsamples in nsamples_list:
            for t in range(5):
                b = []
                seed = 123 + t 
                opt_file = 'fw_log_20_' + str(nsamples) + '_10_0.4_200_' + str(key_list[p]) + '_' + str(seed) + '.txt'
                if opt_file not in file_list:
                    print opt_file
                    sys.exit()

                f = open(solution_dir + opt_file, 'r')
                for i in range(4):
                    obj = float(next(f).split(' ')[1])
                    b.append(obj)
                f.close()
                plt.plot(range(1, 5), b, linestyle = '-')
    plt.xlabel('FW iterations')
    plt.ylabel('Relaxation objective')
    plt.show()


def parse_timing():

    solution_dir = '/home/pankaj/Sampling/data/input/social_graphs/N_512/fw_log/'
    file_list = os.listdir(solution_dir)
    key_list = ['0_0_0.0', '0_0_0.1', '0_1_0.1']
    label_list = ['simple MC', 'convex with fw opt. 0.1', 'convex with greedy opt. 0.1']
    for p in range(3):
        d = []
        e = []
        nsamples_list = [1, 5, 10, 20]
        for nsamples in nsamples_list:
            a = []
            c = []
            for g_id in range(5):
                b = []
                for t in range(5):
                    seed = 123 + t 
                    log_file = 'g_N_512_' + str(g_id) + '_20_' + str(nsamples) + '_20_0.4_100_' + str(key_list[p]) + '_' + str(seed) + '.txt'
                    if log_file not in file_list:
                        log_file = 'g_N_512_' + str(g_id) + '_20_' + str(nsamples) + '_10_0.4_100_' + str(key_list[p]) + '_' + str(seed) + '.txt'
                        if log_file not in file_list:
                            print log_file
                            sys.exit()

                    f = open(solution_dir + log_file, 'r')
                    for i in range(9):
                        next(f)
                    timing = float(next(f).split(' ')[0])
                    f.close()
                    b.append(timing)
                c.append(np.mean(b))
            e.append(np.mean(c))
        print e
 
# Gather our code in a main() function
def main():
     filename = '/home/pankaj/Sampling/data/input/social_graphs/N_512/fw_log/' + sys.argv[1]
     save_filename = sys.argv[2]
     f = open(filename, 'r')
     val = []
     for line in f:
	print line
 	a = line.split(' ')
	val.append(float(a[1]))
     linePlotSave(val, save_filename)

if __name__ == '__main__':
#    main()
#    plot_iterates_hist()
#    plot_email_solution_variance()
#    plot_email_obj()
    plot_facebook_obj()
#    parse_email_timing()
