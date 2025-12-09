import numpy as np
import torch
import byzantine_pytorch as byzantine
import wandb
from sklearn.metrics import roc_auc_score
import hdbscan
import torch.nn.functional as F

def block_wise_median(param_values):
    sorted_values = torch.sort(param_values, dim=-1)[0]
    return sorted_values[:, param_values.shape[-1] // 2]

def block_wise_trim(param_values, b, m):
    sorted_values = torch.sort(param_values, dim=-1)[0]
    return sorted_values[:, b:(b+m)].mean(dim=-1)


def cos_sim_nd(p, q):
    return 1 - (p * q / (torch.norm(p) * torch.norm(q))).sum()



# median
def median(gradients, net, lr, nfake, byz, history,  fixed_rand, init_model, last_50_model, last_grad, sf, e):
    # Get device from net
    device = next(net.parameters()).device
    # Convert to torch tensors
    param_list = [torch.cat([xx.reshape(-1, 1) for xx in x], dim=0).to(device) for x in gradients]
    """if byz == byzantine.fang_attack or byz == byzantine.opt_fang:
        param_list, sf = byz(param_list, net, lr, nfake, history,
                          fixed_rand,  init_model, last_50_model, last_grad, e, sf, "median")

    else:"""
    param_list, sf = byz(param_list, net, lr, nfake,
                        history,  fixed_rand,  init_model, last_50_model, last_grad,e, sf)
    for i, param in enumerate(param_list):
        mask = torch.isnan(param) | torch.isinf(param)
        param_list[i] = torch.where(mask, torch.ones_like(param)*100000, param)

    if len(param_list) >= 100:
        block_size = 10000
        block_wise_nd = torch.cat(param_list, dim=1)
        global_update = torch.zeros(param_list[0].numel(), device=device)
        for i in range(global_update.numel() // block_size):
            global_update[i * block_size : (i + 1) * block_size] = block_wise_median(block_wise_nd[i * block_size : (i + 1) * block_size])
        global_update[global_update.numel() // block_size * block_size : global_update.numel()] = block_wise_median(block_wise_nd[global_update.numel() // block_size * block_size : global_update.numel()])
    else:
        sorted_array = torch.sort(torch.cat(param_list, dim=1), dim=-1)[0]
        if sorted_array.shape[-1] % 2 == 1:
            global_update = sorted_array[:, int(sorted_array.shape[-1] / 2)]
        else:
            global_update = (sorted_array[:, int((sorted_array.shape[-1] / 2 - 1))] + sorted_array[:, int((sorted_array.shape[-1] / 2))]) / 2

    # update the global model using PyTorch
    idx = 0
    with torch.no_grad():
        for j, param in enumerate(net.parameters()):
            param.add_(global_update[idx:(idx+param.numel())].reshape(param.shape))
            idx += param.numel()
    return param_list, sf

def krum(gradients, net, lr, nfake, byz, history, fixed_rand, init_model, last_50_model, last_grad, sf, e):
    device = next(net.parameters()).device
    # Flatten gradients into a list of (d, 1) tensors
    param_list = [torch.cat([xx.reshape(-1, 1) for xx in x], dim=0).to(device) for x in gradients]
    
    # Apply attack
    if byz == byzantine.fang_attack or byz == byzantine.opt_fang:
        param_list, sf = byz(param_list, net, lr, nfake, history,
                          fixed_rand,  init_model, last_50_model, last_grad, e, sf, "krum")
    else:
        param_list, sf = byz(param_list, net, lr, nfake, history, fixed_rand, init_model, last_50_model, last_grad, e, sf)

    # Stack all updates to (d, n)
    v = torch.cat(param_list, dim=1)
    if byz == byzantine.no_byz:
        v = v[:, nfake:]  # Exclude attackers for no_byz case
    n = v.shape[1]
    f = nfake
    # Krum parameter: sum of distances to n - f - 2 neighbors
    k = n - f - 2
    
    # Compute pairwise Euclidean distances (n, n)
    # v.T is (n, d), result is (n, n)
    dists = torch.cdist(v.T, v.T)
    
    # Find k+1 smallest distances (including self-distance which is 0)
    # largest=False means smallest
    topk_dists, _ = torch.topk(dists, k + 1, dim=1, largest=False)
    
    # Krum score is sum of squared distances
    scores = torch.sum(topk_dists**2, dim=1)
    
    # Select the client with the minimum score
    idx = torch.argmin(scores)
    global_update = v[:, idx]
    
    # Update global model
    idx_p = 0
    with torch.no_grad():
        for j, param in enumerate(net.parameters()):
            param.add_(global_update[idx_p:(idx_p+param.numel())].reshape(param.shape))
            idx_p += param.numel()
            
    return param_list, sf


def multi_krum(gradients, net, lr, nfake, byz, history, fixed_rand, init_model, last_50_model, last_grad, sf, e):
    device = next(net.parameters()).device
    param_list = [torch.cat([xx.reshape(-1, 1) for xx in x], dim=0).to(device) for x in gradients]
    
    # Apply attack
    if byz == byzantine.fang_attack or byz == byzantine.opt_fang:
        param_list, sf = byz(param_list, net, lr, nfake, history,
                          fixed_rand,  init_model, last_50_model, last_grad, e, sf, "krum")
    else:
        param_list, sf = byz(param_list, net, lr, nfake, history, fixed_rand, init_model, last_50_model, last_grad, e, sf)
    
    v = torch.cat(param_list, dim=1)
    if byz == byzantine.no_byz:
        v = v[:, nfake:]  # Exclude attackers for no_byz case
    n = v.shape[1]
    f = nfake
    k = n - f - 2
    # Multi-Krum selects 'm' benign clients (usually n - f)
    m = n - f
    
    # Compute scores similar to Krum
    dists = torch.cdist(v.T, v.T)
    topk_dists, _ = torch.topk(dists, k + 1, dim=1, largest=False)
    scores = torch.sum(topk_dists**2, dim=1)
    
    # Select top m clients with smallest scores
    _, indices = torch.topk(scores, m, largest=False)
    
    # Compute mean of selected updates
    selected_updates = v[:, indices]
    global_update = torch.mean(selected_updates, dim=1)
    
    # Update global model
    idx_p = 0
    with torch.no_grad():
        for j, param in enumerate(net.parameters()):
            param.add_(global_update[idx_p:(idx_p+param.numel())].reshape(param.shape))
            idx_p += param.numel()
            
    return param_list, sf

# mean
def simple_mean(gradients, net, lr, nfake, byz, history, fixed_rand,  init_model, last_50_model, last_grad, sf, e):
    device = next(net.parameters()).device
    param_list = [torch.cat([xx.reshape(-1, 1) for xx in x], dim=0).to(device) for x in gradients]
    param_list, sf = byz(param_list, net, lr, nfake, history, fixed_rand,  init_model, last_50_model, last_grad,e, sf)
    # update the global model
    global_update = torch.mean(torch.cat(param_list, dim=1), dim=-1)
    idx = 0
    with torch.no_grad():
        for j, param in enumerate(net.parameters()):
            param.add_(global_update[idx:(idx+param.numel())].reshape(param.shape))
            idx += param.numel()
    return param_list, sf
        
def mean_norm(gradients, net, lr, nfake, byz, history, fixed_rand,  init_model, last_50_model, last_grad, sf, e):
    device = next(net.parameters()).device
    param_list = [torch.cat([xx.reshape(-1, 1) for xx in x], dim=0).to(device) for x in gradients]
    param_list, sf = byz(param_list, net, lr, nfake, history, fixed_rand,  init_model, last_50_model, last_grad,e, sf)

    # update the global model
    param_list = torch.cat(param_list, dim=1)
    param_norms = torch.norm(param_list, dim=0, keepdim=True)
    nb = torch.sum(param_norms[0,nfake:])/(len(param_norms[0])-nfake)
    param_list = param_list * torch.minimum(param_norms + 1e-7, nb) / (param_norms + 1e-7)
    global_update = torch.mean(param_list, dim=-1)
    idx = 0
    with torch.no_grad():
        for j, param in enumerate(net.parameters()):
            param.add_(global_update[idx:(idx+param.numel())].reshape(param.shape))
            idx += param.numel()
    return param_list, sf
    
def score(gradient, v, nfake):
    num_neighbours = v.shape[1] - 2 - nfake
    distances = torch.square(v - gradient).sum(dim=0)
    sorted_distance = torch.sort(distances)[0]
    return torch.sum(sorted_distance[1:(1+num_neighbours)]).item()

def nearest_distance(gradient, c_p):
    distances = torch.square(c_p - gradient).sum(dim=1)
    sorted_distance = torch.sort(distances, dim=0)[0]
    return sorted_distance[1].item()

def score_gmm(gradient, v, nfake):
    num_neighbours = nfake - 1
    distances = torch.square(v - gradient).sum(dim=0)
    sorted_distance = torch.sort(distances)[0]
    return torch.sum(sorted_distance[1:(1+num_neighbours)]).item()
 



# trimmed mean        
def trim(gradients, net, lr, nfake, byz, history,  fixed_rand,  init_model, last_50_model, last_grad, sf, e):
    device = next(net.parameters()).device
    param_list = [torch.cat([xx.reshape(-1, 1) for xx in x], dim=0).to(device) for x in gradients]
    # if byz == byzantine.fang_attack or byz == byzantine.opt_fang: 
    #     param_list, sf = byz(param_list, net, lr, nfake, history, fixed_rand,  init_model, last_50_model, last_grad,e, sf, "trim")
    # else: 
    param_list, sf = byz(param_list, net, lr, nfake, history,  fixed_rand,  init_model, last_50_model, last_grad,e, sf)
    b = nfake
    n = len(param_list)
    m = n - b*2
    for i, param in enumerate(param_list):
        mask = torch.isnan(param) | torch.isinf(param)
        param_list[i] = torch.where(mask, torch.ones_like(param)*100000, param)

    if m <= 0:
        return -1
    
    if len(param_list) >= 100:
        block_size = 10000
        block_wise_nd = torch.cat(param_list, dim=1)
        global_update = torch.zeros(param_list[0].numel(), device=device)
        for i in range(global_update.numel() // block_size):
            global_update[i * block_size : (i + 1) * block_size] = block_wise_trim(block_wise_nd[i * block_size : (i + 1) * block_size], b, m)
        global_update[global_update.numel() // block_size * block_size : global_update.numel()] = block_wise_trim(block_wise_nd[global_update.numel() // block_size * block_size : global_update.numel()], b, m)
    
    else:
        sorted_array = torch.sort(torch.cat(param_list, dim=1), dim=-1)[0]
        global_update = torch.mean(sorted_array[:, b:(b+m)], dim=-1)

    # update the global model
    idx = 0
    with torch.no_grad():
        for j, param in enumerate(net.parameters()):
            param.add_(global_update[idx:(idx+param.numel())].reshape(param.shape))
            idx += param.numel()
    
    return param_list, sf


def fltrust(gradients, net, lr, nfake, byz, history, fixed_rand, init_model, last_50_model, last_grad, sf, e, server_gradient=None):
    """
    FLTrust aggregation: Reweighting by cosine similarity and rescaling by gradient norm.
    """
    if server_gradient is None:
        raise ValueError("Server gradient required for FLTrust")

    device = next(net.parameters()).device
    param_list = [torch.cat([xx.reshape(-1, 1) for xx in x], dim=0).to(device) for x in gradients]
    param_list, sf = byz(param_list, net, lr, nfake, history, fixed_rand,  init_model, last_50_model, last_grad, e, sf)
    for i, param in enumerate(param_list):
        mask = torch.isnan(param) | torch.isinf(param)
        param_list[i] = torch.where(mask, torch.ones_like(param)*100000, param)
    
    # Flatten server gradient (g0)
    g0 = torch.cat([x.reshape(-1, 1) for x in server_gradient]).to(device)
    mask = torch.isnan(g0) | torch.isinf(g0)
    g0 = torch.where(mask, torch.ones_like(g0)*100000, g0)
    g_clients = torch.cat(param_list, dim=1)  # Shape: (d, n_clients)

    cos_sim = F.cosine_similarity(g0, g_clients, dim=0)  # Shape: (n_clients,)
    bias = -0.1
    # trust_score = torch.relu((cos_sim + bias) / (1 + bias)) # Shape: (n_clients,)
    # trust_score = torch.where(cos_sim > 0.1, cos_sim, torch.zeros_like(cos_sim))
    trust_score = torch.where(cos_sim > 0.1, torch.ones_like(cos_sim), torch.zeros_like(cos_sim))
    print("FLTrust trust_score:", trust_score)
    print(f"attacker's total trust score: {trust_score[:nfake].sum().item():.4f}, benign's total trust score: {trust_score[nfake:].sum().item():.4f}")

    # 2. Compute Rescaling Factor: score * (norm_g0 / norm_gi)
    # FLTrust scales client gradient to have same magnitude as server gradient
    scaling = trust_score / (trust_score.sum() + 1e-9)
    print(f"attacker's total scaling: {scaling[:nfake].sum().item():.4f}, benign's total scaling: {scaling[nfake:].sum().item():.4f}")

    # 3. Weighted Average
    # Formula: sum(scaling_i * g_i) / sum(trust_score_i)
    global_update = torch.sum(g_clients * scaling.unsqueeze(0), dim=1)

    # Reshape back to model parameters
    idx = 0
    with torch.no_grad():
        for j, param in enumerate(net.parameters()):
            param.add_(global_update[idx:(idx+param.numel())].reshape(param.shape))
            idx += param.numel()
    return param_list, sf


def flatten_and_stack_client_updates(client_update_lists):
    """
    將多層次的客戶端更新列表，轉換為 SpecGuard 所需的 (M, d) 矩陣。

    Args:
        client_update_lists (list): 結構為 [ [客戶端1的參數更新張量列表], ... ]

    Returns:
        torch.Tensor: 包含所有客戶端更新的矩陣 G_client (M, d)。
    """
    G_client_rows = [] # 用來收集 M 個 (1, d) 的行向量

    for client_tensors in client_update_lists:
        # 1. 內層操作：將單一客戶端的所有 N_params 個 Tensor 串接成 d 維向量

        # 將每個參數 Tensor 拉平為 1D 向量
        flattened_tensors = [t.reshape(-1) for t in client_tensors]
        
        # 串接所有的拉平向量，得到一個 (d,) 的向量
        client_d_vector = torch.cat(flattened_tensors, dim=0)
        
        # 轉換為 (1, d) 的行向量，準備堆疊
        client_row_vector = client_d_vector.reshape(1, -1)
        
        G_client_rows.append(client_row_vector)

    # 2. 外層操作：將 M 個 (1, d) 的行向量堆疊成最終矩陣
    
    # 檢查是否為空，避免 torch.cat 出錯
    if not G_client_rows:
        return None 
        
    G_client = torch.cat(G_client_rows, dim=0) # Shape: (M, d)
    
    return G_client


def specguard(gradients, net, lr, nfake, byz, history, fixed_rand,  init_model, last_50_model, last_grad, sf, e, V_ref):
    device = next(net.parameters()).device
    param_list = [torch.cat([xx.reshape(-1, 1) for xx in x], dim=0).to(device) for x in gradients]
    """if byz == byzantine.fang_attack or byz == byzantine.opt_fang:
        param_list, sf = byz(param_list, net, lr, nfake, history,
                          fixed_rand,  init_model, last_50_model, last_grad, e, sf, "median")   
    else:"""
    param_list, sf = byz(param_list, net, lr, nfake,
                        history,  fixed_rand,  init_model, last_50_model, last_grad, e, sf)

    for i, param in enumerate(param_list):
        mask = torch.isnan(param) | torch.isinf(param)
        assert mask.sum() < 1
        param_list[i] = torch.where(mask, torch.ones_like(param)*100000, param)
        #param_list[i] = torch.where(mask, torch.zeros_like(param), param)
    
    # for i in range(10):
    #     print(param_list[i].sum().item(),param_list[i].abs().sum().item())
    
    # Ensure V_ref is a torch tensor on the same device
    if not isinstance(V_ref, torch.Tensor):
        V_ref = torch.from_numpy(V_ref).to(device)
    elif V_ref.device != device:
        V_ref = V_ref.to(device)
    
    # print('V_ref shape', V_ref.shape)
    # print(len(param_list),len(param_list[0]),len(param_list[0][0]))
    # need to flatten param_list to be (num_clients, d)

    #G_client = flatten_and_stack_client_updates(param_list)  # Shape: (num_clients, d)
    G_client = torch.cat(param_list, dim=1).T # Shape: (num_clients, d)
    # print('flattened_param_list shape', G_client.shape)
    # if G_client is (num,d,1), convert to (num,d)
    # G_client = G_client.reshape((G_client.shape[0], G_client.shape[1]))
    
    # GPU-accelerated matrix multiplication
    projection_matrix = torch.mm(V_ref, G_client.T) # [5, 36]
    E_signal = torch.sum(projection_matrix*projection_matrix, dim=0) # [36] clients proj_sum^2
    E_sum = torch.sum(G_client*G_client, dim=1) # [36] clients length^2
    R_scores = E_signal / (E_sum + 1e-8) 
    # print(R_scores)
    # threshold = 0.1 # 閾值，需實驗調優
    # mask_retained = R_scores >= threshold
    # retained_indices = torch.nonzero(mask_retained).squeeze(-1).tolist()
    #_, indeces = torch.topk(R_scores, k=int(G_client.shape[0]*0.25), largest=True)

    # not use topk but use middle 50%
    sorted_R, indeces = torch.sort(R_scores, descending=False)
    lower_bound = int(G_client.shape[0]*0.25)
    upper_bound = int(G_client.shape[0]*0.75)
    indeces = indeces[lower_bound:upper_bound]

    print("SpecGuard R_scores:", R_scores)
    print("indeces", indeces)
    retained_indices = indeces.squeeze(-1).tolist()

    ###
    cnt = 0
    for i in retained_indices:
        if i < nfake:
            cnt += 1
    print(f"SpecGuard retained {len(retained_indices)} clients, including {cnt} attackers, ratio {cnt/(len(retained_indices)+1e-8):.2f}")
    wandb.log({"defense/retained_attacker_ratio": cnt/(len(retained_indices)+1e-8)})
    ###

    # use median to aggregate the retained gradients
    retained_param_list = [param_list[i] for i in retained_indices]
    if len(retained_param_list) == 0:
        print("No reliable clients detected, skipping aggregation.")
        return param_list, sf,0
    
    sorted_array = torch.sort(torch.cat(retained_param_list, dim=1), dim=-1)[0]
    if sorted_array.shape[-1] % 2 == 1:
        global_update = sorted_array[:, int(sorted_array.shape[-1] / 2)]
    else:
        global_update = (sorted_array[:, int((sorted_array.shape[-1] / 2 - 1))] + sorted_array[:, int((sorted_array.shape[-1] / 2))]) / 2   
    # update the global model
    idx = 0
    with torch.no_grad():
        for j, param in enumerate(net.parameters()):
            param.add_(global_update[idx:(idx+param.numel())].reshape(param.shape))
            idx += param.numel()
    return param_list, sf,len(retained_param_list)

def specguard2(gradients, net, lr, nfake, byz, history, fixed_rand,  init_model, last_50_model, last_grad, sf, e):
    device = next(net.parameters()).device
    param_list = [torch.cat([xx.reshape(-1, 1) for xx in x], dim=0).to(device) for x in gradients]
    """if byz == byzantine.fang_attack or byz == byzantine.opt_fang:
        param_list, sf = byz(param_list, net, lr, nfake, history,
                          fixed_rand,  init_model, last_50_model, last_grad, e, sf, "median")   
    else:"""
    param_list, sf = byz(param_list, net, lr, nfake,
                        history,  fixed_rand,  init_model, last_50_model, last_grad, e, sf)

    for i, param in enumerate(param_list):
        mask = torch.isnan(param) | torch.isinf(param)
        assert mask.sum() < 1
        param_list[i] = torch.where(mask, torch.ones_like(param)*100000, param)
    
    # for i in range(10):
    #     print(param_list[i].sum().item(),param_list[i].abs().sum().item())
    
    # Ensure V_ref is a torch tensor on the same device
    
    # print('V_ref shape', V_ref.shape)
    # print(len(param_list),len(param_list[0]),len(param_list[0][0]))
    # need to flatten param_list to be (num_clients, d)

    #G_client = flatten_and_stack_client_updates(param_list)  # Shape: (num_clients, d)
    G_client = torch.cat(param_list, dim=1).T # Shape: (num_clients, d)

    centered_G = G_client - G_client.mean(dim=0, keepdim=True)
    # compute V_ref via SVD on the G_client
    U, S, Vh = torch.linalg.svd(centered_G, full_matrices=False)
    V_ref = Vh[:5, :].to(device)  # Take top 5 right singular vectors

    print('V_ref shape', V_ref.shape)

    # print('flattened_param_list shape', G_client.shape)
    # if G_client is (num,d,1), convert to (num,d)
    # G_client = G_client.reshape((G_client.shape[0], G_client.shape[1]))
    
    # GPU-accelerated matrix multiplication
    projection_matrix = torch.mm(V_ref, G_client.T) # [5, 36]
    E_signal = torch.sum(projection_matrix*projection_matrix, dim=0) # [36] clients proj_sum^2
    E_sum = torch.sum(G_client*G_client, dim=1) # [36] clients length^2
    R_scores = E_signal / (E_sum + 1e-8) 
    # print(R_scores)
    # threshold = 0.1 # 閾值，需實驗調優
    # mask_retained = R_scores >= threshold
    # retained_indices = torch.nonzero(mask_retained).squeeze(-1).tolist()
    #_, indeces = torch.topk(R_scores, k=int(G_client.shape[0]*0.25), largest=True)
    sorted_R, indeces = torch.sort(R_scores, descending=False)
    lower_bound = int(G_client.shape[0]*0.25)
    upper_bound = int(G_client.shape[0]*0.75)
    indeces = indeces[lower_bound:upper_bound]

    print("SpecGuard R_scores:", R_scores)
    print("indeces", indeces)
    retained_indices = indeces.squeeze(-1).tolist()

    ###
    cnt = 0
    for i in retained_indices:
        if i < nfake:
            cnt += 1
    print(f"SpecGuard retained {len(retained_indices)} clients, including {cnt} attackers, ratio {cnt/(len(retained_indices)+1e-8):.2f}")
    wandb.log({"defense/retained_attacker_ratio": cnt/(len(retained_indices)+1e-8)})
    ###

    # use median to aggregate the retained gradients
    retained_param_list = [param_list[i] for i in retained_indices]
    if len(retained_param_list) == 0:
        print("No reliable clients detected, skipping aggregation.")
        return param_list, sf,0
    
    sorted_array = torch.sort(torch.cat(retained_param_list, dim=1), dim=-1)[0]
    if sorted_array.shape[-1] % 2 == 1:
        global_update = sorted_array[:, int(sorted_array.shape[-1] / 2)]
    else:
        global_update = (sorted_array[:, int((sorted_array.shape[-1] / 2 - 1))] + sorted_array[:, int((sorted_array.shape[-1] / 2))]) / 2   
    # update the global model
    idx = 0
    with torch.no_grad():
        for j, param in enumerate(net.parameters()):
            param.add_(global_update[idx:(idx+param.numel())].reshape(param.shape))
            idx += param.numel()
    return param_list, sf,len(retained_param_list)
