def getCondGrad(grad):
    n = grad.shape[0]
    y = Variable(torch.zeros(n)) #conditional grad
    for p in range(n):
        if grad.data[p] > 0:
            y[p] = 1
        else:
            y[p] = 0
    return y

def grad_importance(L, x, net, use_net, nsamples):
#Returns the gradient vector of the multilinear relaxation at x as given in Chekuri's paper
#(See Theorem 1 in nips2012 paper)
    N = L.shape[0]
    grad = Variable(torch.zeros(N))
    input_sample = torch.cat((x, L.view(1, N*N)), dim = 1)
    if use_net == 1:
#        z = net(input_sample, 0) 
        z = net(x, 0) 
    else:
        z = x.clone()

#    if use_net == 1:
#        print x, z
    for p in np.arange(N):
        z_include = z.clone()
        z_exclude  = z.clone() 
        z_include[:, p] = 1
        z_exclude[:, p] = 0

        x_include = x.clone()
        x_exclude  = x.clone() 
        x_include[:, p] = 1
        x_exclude[:, p] = 0

        grad[p] =  multilinear_importance(x_include, z_include, L, nsamples) - multilinear_importance(x_exclude, z_exclude, L, nsamples)

    return grad


def runFrankWolfe(graph_file, nsamples):

    N = L.shape[0]
    x = Variable(torch.Tensor(1, N))
    X = Variable(torch.Tensor(1, (N + 1)*N))
    x[0] = Variable(torch.Tensor([0.5]*N))

    for iter_num in np.arange(1, 50):

        grad = getGrad(L, x, net, use_net, nsamples)
        x_star = getCondGrad(grad)

        step = 2.0/(iter_num + 2) 

        x = step*x_star + (1 - step)*x
    
    currentVal = getExactRelax(L, x)
    return x, currentVal.data[0]

