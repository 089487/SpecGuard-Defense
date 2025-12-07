import numpy as np
import torch

def no_byz(v, net, lr, nfake, history, fixed_rand, init_model, last_50_model, last_grad, e, scaling_factor=100000.):
    return v, scaling_factor


def compute_lambda(all_updates, model_re, n_attackers):
    """Compute lambda for AGR-tailored attacks"""
    distances = []
    n_benign, d = all_updates.shape
    for update in all_updates:
        distance = torch.norm(all_updates - update, dim=1)
        distances.append(distance)
    distances = torch.stack(distances)

    distances, _ = torch.sort(distances, dim=1)
    scores = torch.sum(distances[:, :n_benign - 1 - n_attackers], dim=1)
    min_score = torch.min(scores)
    term_1 = min_score / ((n_benign - n_attackers - 1)
                          * torch.sqrt(torch.tensor([float(d)], device=all_updates.device))[0])
    max_wre_dist = torch.max(torch.norm(all_updates - model_re,
                          dim=1)) / (torch.sqrt(torch.tensor([float(d)], device=all_updates.device))[0])
    return (term_1 + max_wre_dist)


def score(gradient, v, nbyz):
    num_neighbours = v.shape[0] - 2 - nbyz
    sorted_distance, _ = torch.sort(torch.square(v - gradient).sum(dim=1))
    return torch.sum(sorted_distance[1:(1+num_neighbours)]).item()


def poisonedfl(v, net, lr, nfake, history, fixed_rand, init_model, last_50_model, last_grad, e, scaling_factor=100000., gamma=0):
    # k_99 and k_95 for binomial variable for different d for different networks
    if fixed_rand.shape[0] == 1204682:
        k_95 = 603244
        k_99 = 603618
    elif fixed_rand.shape[0] == 139960:
        k_95 = 70288
        k_99 = 70415
    elif fixed_rand.shape[0] == 717924:
        k_95 = 359659
        k_99 = 359948
    elif fixed_rand.shape[0] == 145212:
        k_95 = 72919
        k_99 = 73049
    else:
        raise NotImplementedError
    sf = scaling_factor

    # Start from round 2
    if isinstance(history, torch.Tensor):
        # calculate unit scale vector
        current_model = [param.data.clone() for param in net.parameters()]
        history_norm = torch.norm(history)
        last_grad_norm = torch.norm(last_grad) # scale: v (unnormalized)
        # scale = torch.norm(history - torch.unsqueeze(last_grad, dim=-1) * history_norm / (last_grad_norm + 1e-9), dim=1)
        # scale = torch.abs(history - torch.unsqueeze(last_grad, dim=-1) * history_norm / (last_grad_norm + 1e-9)).squeeze()
        scale = torch.abs(torch.squeeze(history) - last_grad * history_norm / (last_grad_norm + 1e-9))
        deviation = scale * fixed_rand / (torch.norm(scale) + 1e-9) # v@s (g=lambda*deviation)
        
        # calculate scaling factor lambda
        if e % 50 == 0:
            total_update = torch.cat([xx.reshape((-1, 1)) for xx in current_model],
                                dim=0) - torch.cat([xx.reshape((-1, 1)) for xx in last_50_model], dim=0)
            total_update = torch.where(total_update == 0, torch.cat([xx.reshape((-1, 1)) for xx in current_model], dim=0), total_update)
            current_sign = torch.sign(total_update)
            aligned_dim_cnt = (current_sign == torch.unsqueeze(fixed_rand, dim=-1)).sum() 
            if aligned_dim_cnt < k_99 and scaling_factor * 0.7 >= 0.5:
                sf = scaling_factor * 0.7
            else:
                sf = scaling_factor
            lamda_succ = sf * history_norm
        else:
            sf = scaling_factor
            lamda_succ = sf * history_norm
        mal_update = lamda_succ * deviation
        print(sf, lamda_succ, history_norm)
        print("mal_update", mal_update)
        for i in range(nfake):
            epsilon = torch.randn(mal_update.shape, device=mal_update.device)
            noise = gamma * epsilon * lamda_succ / (torch.norm(epsilon) + 1e-9)
            v[i] = torch.unsqueeze(mal_update + noise, dim=-1)
            # v[i] = torch.unsqueeze(mal_update, dim=-1)
    return v, sf


def lie_attack(v, net, lr, nfake, history, fixed_rand, init_model, last_50_model, last_grad, e, scaling_factor=100000.):
    """A Little Is Enough attack"""
    if len(v) <= nfake:
        return v, scaling_factor
    
    # Compute mean and std of benign updates
    benign_updates = torch.stack([v[i].squeeze() for i in range(nfake, len(v))], dim=0)
    mean_update = torch.mean(benign_updates, dim=0)
    std_update = torch.std(benign_updates, dim=0)
    
    # Craft malicious updates: mean - z * std
    z = 3.0  # Standard deviation multiplier
    mal_update = mean_update - z * std_update
    
    for i in range(nfake):
        v[i] = mal_update.unsqueeze(-1)
    
    return v, scaling_factor


def fang_attack(v, net, lr, nfake, history, fixed_rand, init_model, last_50_model, last_grad, e, scaling_factor=100000., agg_type="trim"):
    """AGR-tailored attack (Fang et al.)"""
    if len(v) <= nfake:
        return v, scaling_factor
    
    # Compute benign updates
    benign_updates = torch.cat([v[i] for i in range(nfake, len(v))], dim=1)
    
    # Compute model update
    current_model = [param.data.clone() for param in net.parameters()]
    model_update = torch.cat([xx.reshape(-1, 1) for xx in current_model], dim=0)
    
    # Compute lambda
    if agg_type == "trim":
        # For trimmed mean, compute deviation
        mean_update = torch.mean(benign_updates, dim=1, keepdim=True)
        deviations = torch.norm(benign_updates - mean_update, dim=0)
        max_deviation = torch.max(deviations)
        lamda = max_deviation.item()
    else:  # median
        # For median, use smaller perturbation
        std_update = torch.std(benign_updates, dim=1)
        lamda = torch.mean(std_update).item()
    
    # Craft malicious update
    mean_update = torch.mean(benign_updates, dim=1, keepdim=True)
    direction = mean_update / (torch.norm(mean_update) + 1e-9)
    mal_update = mean_update + lamda * direction
    
    for i in range(nfake):
        v[i] = mal_update
    
    return v, scaling_factor


def opt_fang(v, net, lr, nfake, history, fixed_rand, init_model, last_50_model, last_grad, e, scaling_factor=100000., agg_type="trim"):
    """Optimized AGR-tailored attack"""
    return fang_attack(v, net, lr, nfake, history, fixed_rand, init_model, last_50_model, last_grad, e, scaling_factor, agg_type)


def min_max(v, net, lr, nfake, history, fixed_rand, init_model, last_50_model, last_grad, e, scaling_factor=100000.):
    """Min-Max attack"""
    if len(v) <= nfake:
        return v, scaling_factor
    
    # Compute benign updates
    benign_updates = torch.stack([v[i].squeeze() for i in range(nfake, len(v))], dim=0)
    
    # Find the update with maximum norm
    norms = torch.norm(benign_updates, dim=1)
    max_idx = torch.argmax(norms)
    max_update = benign_updates[max_idx]
    
    # Flip the sign and amplify
    mal_update = -scaling_factor * max_update / (torch.norm(max_update) + 1e-9)
    
    for i in range(nfake):
        v[i] = mal_update.unsqueeze(-1)
    
    return v, scaling_factor


def min_sum(v, net, lr, nfake, history, fixed_rand, init_model, last_50_model, last_grad, e, scaling_factor=100000.):
    """Min-Sum attack"""
    if len(v) <= nfake:
        return v, scaling_factor
    
    # Compute benign updates
    benign_updates = torch.stack([v[i].squeeze() for i in range(nfake, len(v))], dim=0)
    
    # Compute mean of benign updates
    mean_update = torch.mean(benign_updates, dim=0)
    
    # Flip and scale
    mal_update = -scaling_factor * mean_update / (torch.norm(mean_update) + 1e-9)
    
    for i in range(nfake):
        v[i] = mal_update.unsqueeze(-1)
    
    return v, scaling_factor


def random_attack(v, net, lr, nfake, history, fixed_rand, init_model, last_50_model, last_grad, e, scaling_factor=100000.):
    """Random Gaussian noise attack"""
    device = v[0].device if isinstance(v[0], torch.Tensor) else torch.device('cpu')
    for i in range(nfake):
        v[i] = scaling_factor * torch.randn(v[0].shape, device=device)
    return v, scaling_factor


def init_attack(v, net, lr, nfake, history, fixed_rand, init_model, last_50_model, last_grad, e, scaling_factor=100000.):
    """Attack towards initial model"""
    current_model = [param.data.clone() for param in net.parameters()]
    direction = torch.cat([xx.reshape((-1, 1)) for xx in init_model], dim=0) - torch.cat([xx.reshape((-1, 1)) for xx in current_model], dim=0)
    for i in range(nfake):
        v[i] = scaling_factor * direction
    return v, scaling_factor

