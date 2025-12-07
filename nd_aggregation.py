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
    """if byz == byzantine.fang_attack or byz == byzantine.opt_fang:
        param_list, sf = byz(param_list, net, lr, nfake, history,
                          fixed_rand,  init_model, last_50_model, last_grad, e, sf, "median")

    else:"""
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

def flatten_and_stack_client_updates(client_update_lists):
    """
    將多層次的客戶端更新列表，轉換為 SpecGuard 所需的 (M, d) 矩陣。

    Args:
        client_update_lists (list): 結構為 [ [客戶端1的參數更新張量列表], ... ]

    Returns:
        mxnet.ndarray: 包含所有客戶端更新的矩陣 G_client (M, d)。
    """
    G_client_rows = [] # 用來收集 M 個 (1, d) 的行向量

    for client_tensors in client_update_lists:
        # 1. 內層操作：將單一客戶端的所有 N_params 個 Tensor 串接成 d 維向量

        # 將每個參數 Tensor 拉平為 1D 向量
        flattened_tensors = [t.reshape((-1,)) for t in client_tensors]
        
        # 串接所有的拉平向量，得到一個 (d,) 的向量
        client_d_vector = nd.concat(*flattened_tensors, dim=0)
        
        # 轉換為 (1, d) 的行向量，準備堆疊
        client_row_vector = client_d_vector.expand_dims(axis=0) 
        
        G_client_rows.append(client_row_vector)

    # 2. 外層操作：將 M 個 (1, d) 的行向量堆疊成最終矩陣
    
    # 檢查是否為空，避免 nd.concat 出錯
    if not G_client_rows:
        return None 
        
    G_client = nd.concat(*G_client_rows, dim=0) # Shape: (M, d)
    
    return G_client


def specguard(gradients, net, lr, nfake, byz, history, fixed_rand,  init_model, last_50_model, last_grad, sf, e, V_ref):
    param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    """if byz == byzantine.fang_attack or byz == byzantine.opt_fang:
        param_list, sf = byz(param_list, net, lr, nfake, history,
                          fixed_rand,  init_model, last_50_model, last_grad, e, sf, "median")   
    else:"""
    param_list, sf = byz(param_list, net, lr, nfake,
                        history,  fixed_rand,  init_model, last_50_model, last_grad,e, sf)


    for param in param_list:
        mask = mx.nd.contrib.isnan(param) + mx.nd.contrib.isinf(param)
        param = mx.nd.where(mask, mx.nd.ones_like(param)*100000, param) 
    print('V_ref shape', V_ref.shape)
    print(len(param_list),len(param_list[0]),len(param_list[0][0]))
    # need to flatten param_list to be (num_clients, d)

    G_client =nd.concat(*param_list, dim=1).T #flatten_and_stack_client_updates(param_list)  # Shape: (num_clients, d)
    print('flattened_param_list shape', G_client.shape)
    # if G_client is (num,d,1), convert to (num,d)
    G_client = G_client.reshape((G_client.shape[0], G_client.shape[1]))
    
    projection_matrix = nd.dot(V_ref, G_client.T)
    E_signal = nd.sum(projection_matrix*projection_matrix, axis=0)
    E_sum = nd.sum(G_client*G_client, axis=1)
    R_scores = E_signal / E_sum
    print(R_scores)
    threshold = 0.1 # 閾值，需實驗調優
    mask_retained = R_scores >= threshold
    # convert MXNet NDArray boolean mask to a NumPy array and get indices
    retained_indices = np.nonzero(mask_retained.asnumpy())[0].tolist()

    # use median to aggregate the retained gradients
    retained_param_list = [param_list[i] for i in retained_indices]
    if len(retained_param_list) == 0:
        print("No reliable clients detected, skipping aggregation.")
        return param_list, sf,0
    sorted_array = nd.sort(nd.concat(*retained_param_list, dim=1), axis=-1)
    if sorted_array.shape[-1] % 2 == 1:
        global_update = sorted_array[:, int(sorted_array.shape[-1] / 2)]
    else:
        global_update = (sorted_array[:, int((sorted_array.shape[-1] / 2 - 1))] + sorted_array[:, int((sorted_array.shape[-1] / 2))]) / 2   
    # update the global model
    idx = 0
    for j, (param) in enumerate(net.collect_params().values()):
        param.set_data(param.data() +global_update[idx:(idx+param.data().size)].reshape(param.data().shape))
        idx += param.data().size   
    return param_list, sf,len(retained_param_list)



