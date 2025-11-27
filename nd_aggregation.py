import numpy as np
np.bool = np.bool_
import mxnet as mx
from mxnet import nd, autograd, gluon
import byzantine
import wandb
from sklearn.metrics import roc_auc_score
import hdbscan

def block_wise_median(param_values):
    return param_values.sort(axis=-1)[:, param_values.shape[-1] // 2]

def block_wise_trim(param_values, b, m):
    return param_values.sort(axis=-1)[:, b:(b+m)].mean(axis=-1)


def cos_sim_nd(p, q):
    return 1 - (p * q / (p.norm() * q.norm())).sum()



# median
def median(gradients, net, lr, nfake, byz, history,  fixed_rand, init_model, last_50_model, last_grad, sf, e):
    param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    if byz == byzantine.fang_attack or byz == byzantine.opt_fang:
        param_list, sf = byz(param_list, net, lr, nfake, history,
                          fixed_rand,  init_model, last_50_model, last_grad, e, sf, "median")

    else:
        param_list, sf = byz(param_list, net, lr, nfake,
                         history,  fixed_rand,  init_model, last_50_model, last_grad,e, sf)
    for param in param_list:
        mask = mx.nd.contrib.isnan(param) + mx.nd.contrib.isinf(param)
        param = mx.nd.where(mask, mx.nd.ones_like(param)*100000, param)

    if len(param_list) >= 100:
        block_size = 10000
        block_wise_nd = nd.concat(*param_list, dim=1)
        global_update = nd.zeros(param_list[0].size)
        for i in range(global_update.size // block_size):
            global_update[i * block_size : (i + 1) * block_size] = block_wise_median(block_wise_nd[i * block_size : (i + 1) * block_size])
        global_update[global_update.size // block_size * block_size : global_update.size] = block_wise_median(block_wise_nd[global_update.size // block_size * block_size : global_update.size])
    else:
        sorted_array = nd.sort(nd.concat(*param_list, dim=1), axis=-1)
        if sorted_array.shape[-1] % 2 == 1:
            global_update = sorted_array[:, int(sorted_array.shape[-1] / 2)]
        else:
            global_update = (sorted_array[:, int((sorted_array.shape[-1] / 2 - 1))] + sorted_array[:, int((sorted_array.shape[-1] / 2))]) / 2

    global_update.wait_to_read()
    # update the global model
    idx = 0
    for j, (param) in enumerate(net.collect_params().values()):
        param.set_data(param.data() +global_update[idx:(idx+param.data().size)].reshape(param.data().shape))
        idx += param.data().size   
    return param_list, sf

# mean
def simple_mean(gradients, net, lr, nfake, byz, history, fixed_rand,  init_model, last_50_model, last_grad, sf, e):
    param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    param_list, sf = byz(param_list, net, lr, nfake, history, fixed_rand,  init_model, last_50_model, last_grad,e, sf)
    # update the global model
    global_update = nd.mean(nd.concat(*param_list, dim=1), axis=-1)
    idx = 0
    for j, (param) in enumerate(net.collect_params().values()):

        param.set_data(param.data() +global_update[idx:(idx+param.data().size)].reshape(param.data().shape))
        idx += param.data().size   
    return param_list, sf
        
def mean_norm(gradients, net, lr, nfake, byz, history, fixed_rand,  init_model, last_50_model, last_grad, sf, e):
    param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    param_list, sf = byz(param_list, net, lr, nfake, history, fixed_rand,  init_model, last_50_model, last_grad,e, sf)

    # update the global model
    param_list = nd.concat(*param_list, dim=1)
    param_norms = nd.norm(param_list, axis=0, keepdims=True)
    nb = sum(param_norms[0,nfake:])/(len(param_norms[0])-nfake)
    param_list = param_list * nd.minimum(param_norms + 1e-7, nb) / (param_norms+ 1e-7)
    global_update = nd.mean(param_list, axis=-1)
    idx = 0
    for j, (param) in enumerate(net.collect_params().values()):
        param.set_data(param.data() +global_update[idx:(idx+param.data().size)].reshape(param.data().shape))
        idx += param.data().size   
    return param_list, sf
    
def score(gradient, v, nfake):
    num_neighbours = v.shape[1] - 2 - nfake
    sorted_distance = nd.square(v - gradient).sum(axis=0).sort()
    return nd.sum(sorted_distance[1:(1+num_neighbours)]).asscalar()

def nearest_distance(gradient, c_p):
    sorted_distance = nd.square(c_p - gradient).sum(axis=1).sort(axis=0)
    return sorted_distance[1].asscalar()

def score_gmm(gradient, v, nfake):
    num_neighbours = nfake - 1
    sorted_distance = nd.square(v - gradient).sum(axis=0).sort()
    return nd.sum(sorted_distance[1:(1+num_neighbours)]).asscalar()
 



# trimmed mean        
def trim(gradients, net, lr, nfake, byz, history,  fixed_rand,  init_model, last_50_model, last_grad, sf, e):
    param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    if byz == byzantine.fang_attack or byz == byzantine.opt_fang: 
        param_list, sf = byz(param_list, net, lr, nfake, history, fixed_rand,  init_model, last_50_model, last_grad,e, sf, "trim")
    else: 
        param_list, sf = byz(param_list, net, lr, nfake, history,  fixed_rand,  init_model, last_50_model, last_grad,e, sf)
    b = nfake
    n = len(param_list)
    m = n - b*2
    for param in param_list:
        mask = mx.nd.contrib.isnan(param) + mx.nd.contrib.isinf(param)
        param = mx.nd.where(mask, mx.nd.ones_like(param)*100000, param)

    if m <= 0:
        return -1
    
    if len(param_list) >= 100:
        block_size = 10000
        block_wise_nd = nd.concat(*param_list, dim=1)
        global_update = nd.zeros(param_list[0].size)
        for i in range(global_update.size // block_size):
            global_update[i * block_size : (i + 1) * block_size] = block_wise_trim(block_wise_nd[i * block_size : (i + 1) * block_size], b, m)
        global_update[global_update.size // block_size * block_size : global_update.size] = block_wise_trim(block_wise_nd[global_update.size // block_size * block_size : global_update.size], b, m)
    
    else:
        sorted_array = nd.sort(nd.concat(*param_list, dim=1), axis=-1)
        global_update = nd.mean(sorted_array[:, b:(b+m)], axis=-1)

    # update the global model
    idx = 0
    for j, (param) in enumerate(net.collect_params().values()):
        param.set_data(param.data() +global_update[idx:(idx+param.data().size)].reshape(param.data().shape))
        idx += param.data().size  
    
    return param_list, sf
