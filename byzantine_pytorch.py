import numpy as np
import torch
from scipy.stats import norm

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


def poisonedfl(v, net, lr, nfake, history, fixed_rand, init_model, last_50_model, last_grad, e, scaling_factor=100000., gamma=1.2):
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
            noise = gamma * epsilon * fixed_rand * lamda_succ / (torch.norm(epsilon) + 1e-9)
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


def fang_attack(v, net, lr, nfake, history, fixed_rand, init_model, last_50_model, last_grad, e, scaling_factor=100000., agg_type='trim'):
    """
    Original Fang's Attack (Section 4.2 & Algorithm 1 of Fang et al. Usenix Sec 2020)
    """
    if len(v) <= nfake:
        return v, scaling_factor

    # Get benign updates (assuming first nfake are malicious placeholders)
    # v is list of tensors (d, 1)
    benign_updates = torch.cat([v[i] for i in range(nfake, len(v))], dim=1) # (d, n_benign)
    n_benign = benign_updates.shape[1]
    
    # Calculate benign statistics
    benign_mean = torch.mean(benign_updates, dim=1, keepdim=True)
    benign_std = torch.std(benign_updates, dim=1, keepdim=True)
    
    # 1. Attack against Krum (Based on Eq 7-8 and Algorithm 1 in Fang's paper)
    if agg_type == 'krum':
        # Fang uses -sgn(mean) or similar direction
        deviation = torch.sign(benign_mean)
        
        # Compute distances among benign
        dists = torch.cdist(benign_updates.T, benign_updates.T)
        # k for Krum (assuming nfake attackers)
        k = n_benign - 2 # Since we are comparing within benign set effectively? 
        mal_update = -deviation 
        lam = compute_lambda(benign_updates.T, benign_mean.reshape(-1), nfake)
        
        mal_update = benign_mean - lam * deviation
        
    # 2. Attack against Trimmed Mean (Fang's Full Knowledge Attack)
    # Fang sets w_mal to w_max (if mean > 0) or w_min (if mean < 0)
    elif agg_type == 'trim':
        benign_max, _ = torch.max(benign_updates, dim=1, keepdim=True)
        benign_min, _ = torch.min(benign_updates, dim=1, keepdim=True)
        mal_update = torch.where(benign_mean > 0, benign_min, benign_max)
        
    else:
        # Default fallback (e.g. mean)
        mal_update = benign_mean

    # Apply to all malicious clients
    for i in range(nfake):
        v[i] = mal_update
        
    return v, scaling_factor

def solve_krum_lambda_binary_search(benign_updates, nfake, benign_mean, direction):
    """
    Binary search to maximize lambda s.t. Malicious Score <= Min Benign Score
    """
    low = 0.0
    high = 10000.0 # Upper bound
    tol = 1e-4
    n_benign = benign_updates.shape[1]
    n_total = n_benign + nfake
    k = n_total - nfake - 2 # Krum k = n - f - 2
    
    # Pre-compute benign-benign distances (squared)
    # (n_benign, n_benign)
    dist_bb_sq = torch.cdist(benign_updates.T, benign_updates.T, p=2)**2
    
    # Neighbors needed for Malicious node from Benign set
    # Malicious node has (f-1) neighbors with dist 0. Needs k - (f-1) more.
    # If k < f-1, it needs 0 benign neighbors (unlikely in valid Krum setup).
    needed_from_benign = max(0, k - (nfake - 1))
    
    best_lam = 0.0
    
    # Binary Search
    for _ in range(20): # 20 iters is usually enough for precision
        mid = (low + high) / 2
        mal_cand = benign_mean + mid * direction
        
        # 1. Calculate Score of Malicious Candidate
        # Distance from Mal to Benigns
        dist_mb = torch.norm(benign_updates - mal_cand, dim=0) # (n_benign,)
        dist_mb_sq = dist_mb**2
        
        # Get smallest needed_from_benign distances
        if needed_from_benign > 0:
            top_dists, _ = torch.topk(dist_mb_sq, needed_from_benign, largest=False)
            score_mal = torch.sum(top_dists) # Other (f-1) are 0
        else:
            score_mal = torch.tensor(0.0, device=benign_updates.device)

        mal_dists_expanded = dist_mb_sq.unsqueeze(1).expand(n_benign, nfake)
        
        # Concatenate: (n_benign, n_benign + nfake)
        # Note: self-distance is 0, included in topk
        all_dists_for_benign = torch.cat([dist_bb_sq, mal_dists_expanded], dim=1)
        
        # Top k+1 (including self)
        top_k_benign, _ = torch.topk(all_dists_for_benign, k + 1, dim=1, largest=False)
        scores_benign = torch.sum(top_k_benign**2, dim=1)
        
        min_score_benign = torch.min(scores_benign)
        
        # Check Condition
        if score_mal <= min_score_benign:
            best_lam = mid
            low = mid # Try larger
        else:
            high = mid # Too far
            
    return best_lam


def opt_fang(v, net, lr, nfake, history, fixed_rand, init_model, last_50_model, last_grad, e, scaling_factor=100000., agg_type='trim'):
    """
    Optimized AGR-tailored Attack (Shejwalkar et al. NDSS 2021)
    Corrects Fang's suboptimal attacks by solving the optimization problem exactly.
    """
    if len(v) <= nfake:
        return v, scaling_factor

    benign_updates = torch.cat([v[i] for i in range(nfake, len(v))], dim=1) # (d, n_benign)
    benign_mean = torch.mean(benign_updates, dim=1, keepdim=True)
    
    # 1. Optimized Attack against Krum (Binary Search for Gamma)
    if agg_type == 'krum':
        # Direction: Inverse of benign mean (to maximize angular deviation)
        direction = -torch.sign(benign_mean)
        lam = compute_lambda(benign_updates.T, benign_mean.reshape(-1), nfake)
        mal_update = benign_mean + lam * direction
        
    # 2. Optimized Attack against Trimmed Mean (Quantile Attack)
    elif agg_type == 'trim':
        
        # Calculate indices
        n_benign = benign_updates.shape[1]
        n_total = n_benign + nfake
        # Assume Defense removes 'nfake' from head and tail (as per user prompt)
        n_trim = nfake 
        
        # Sort benign values per dimension
        sorted_benign, _ = torch.sort(benign_updates, dim=1) # (d, n_benign)
        
        idx_max = -nfake // 2
        idx_min = nfake // 2
        
        target_max = sorted_benign[:, idx_max].reshape(-1, 1)
        target_min = sorted_benign[:, idx_min].reshape(-1, 1)
        
        mal_update = torch.where(benign_mean > 0, target_min, target_max)
        
    else:
        mal_update = benign_mean

    for i in range(nfake):
        v[i] = mal_update
        
    return v, scaling_factor


def min_max(v, net, lr, nfake, history, fixed_rand, init_model, last_50_model, last_grad, e, scaling_factor=100000.):
    """
    Min-Sum Attack (NDSS 2021)
    Constraint: Sum of Squared Errors (SSE) to all other updates <= Max SSE of benign
    Target: Trimmed Mean, Median, generalized median defenses
    """
    benign_updates = torch.stack(v[nfake:]) # (n_benign, d, 1)
    mu = torch.mean(benign_updates, dim=0, keepdim=True) # (1, d, 1)

    max_sse_benign = torch.max(torch.norm(benign_updates - benign_updates.permute(2, 1, 0), dim=1))
    
    attack_dir = -mu / (torch.norm(mu) + 1e-9)
    
    step = scaling_factor * torch.norm(mu)
    best_gamma = 0.0
    benign_updates = benign_updates - mu
    
    for _ in range(50):
        if torch.norm(benign_updates - attack_dir * (best_gamma + step), dim=1).max() <= max_sse_benign:
            best_gamma += step
        step /= 2.0
        if step < 1e-8:
            break
    for i in range(nfake):
        v[i] = torch.squeeze(mu + attack_dir * best_gamma, dim=0)
        
    return v, scaling_factor

def min_sum(v, net, lr, nfake, history, fixed_rand, init_model, last_50_model, last_grad, e, scaling_factor=100000.):
    """
    Min-Sum Attack (NDSS 2021)
    Constraint: Sum of Squared Errors (SSE) to all other updates <= Max SSE of benign
    Target: Trimmed Mean, Median, generalized median defenses
    """
    benign_updates = torch.stack(v[nfake:]) # (n_benign, d, 1)
    mu = torch.mean(benign_updates, dim=0, keepdim=True) # (1, d, 1)
    
    dist_sums = torch.norm(benign_updates - benign_updates.permute(2, 1, 0), dim=1).sum(dim=0)
        
    max_sse_benign = torch.max(dist_sums)
    
    attack_dir = -mu / (torch.norm(mu) + 1e-9)
    
    step = scaling_factor * torch.norm(mu)
    best_gamma = 0.0
    benign_updates = benign_updates - mu
    
    for _ in range(50):
        if torch.norm(benign_updates - attack_dir * (best_gamma + step), dim=1).sum() <= max_sse_benign:
            best_gamma += step
        step /= 2.0
        if step < 1e-8:
            break
    for i in range(nfake):
        v[i] = torch.squeeze(mu + attack_dir * best_gamma, dim=0)
        
    return v, scaling_factor


def random_attack(v, net, lr, nfake, history, fixed_rand, init_model, last_50_model, last_grad, e, scaling_factor=100000.):
    """Random Gaussian noise attack"""
    device = v[0].device if isinstance(v[0], torch.Tensor) else torch.device('cpu')
    for i in range(nfake):
        noise = torch.randn(v[i].shape, device=device)
        v[i] = scaling_factor * noise / (torch.norm(noise) + 1e-9) * torch.norm(history)
    return v, scaling_factor


def init_attack(v, net, lr, nfake, history, fixed_rand, init_model, last_50_model, last_grad, e, scaling_factor=100000.):
    """Attack towards initial model"""
    current_model = [param.data.clone() for param in net.parameters()]
    direction = torch.cat([xx.reshape((-1, 1)) for xx in init_model], dim=0) - torch.cat([xx.reshape((-1, 1)) for xx in current_model], dim=0)
    direction = direction / (torch.norm(direction) + 1e-9) * torch.norm(history)
    for i in range(nfake):
        v[i] = scaling_factor * direction
    return v, scaling_factor

