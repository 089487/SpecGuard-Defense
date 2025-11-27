import numpy as np
np.bool = np.bool_
from mxnet import nd, autograd, gluon

def no_byz(v, net, lr, nfake, history, fixed_rand, init_model, last_50_model, last_grad, e, scaling_factor=100000.):
    return v, scaling_factor


def compute_lambda(all_updates, model_re, n_attackers):
    distances = []
    n_benign, d = all_updates.shape
    for update in all_updates:
        distance = nd.norm(all_updates - update, axis=1)
        distances.append(distance)
    distances = nd.stack(*distances)

    distances = nd.sort(distances, axis=1)
    scores = nd.sum(distances[:, :n_benign - 1 - n_attackers], axis=1)
    min_score = nd.min(scores)
    term_1 = min_score / ((n_benign - n_attackers - 1)
                          * nd.sqrt(nd.array([d]))[0])
    max_wre_dist = nd.max(nd.norm(all_updates - model_re,
                          axis=1)) / (nd.sqrt(nd.array([d]))[0])
    return (term_1 + max_wre_dist)


def score(gradient, v, nbyz):
    num_neighbours = v.shape[0] - 2 - nbyz
    sorted_distance = nd.square(v - gradient).sum(axis=1).sort()
    return nd.sum(sorted_distance[1:(1+num_neighbours)]).asscalar()


def poisonedfl(v, net, lr, nfake, history, fixed_rand, init_model, last_50_model, last_grad, e, scaling_factor=100000.):
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
    if isinstance(history, nd.NDArray):
        
        # calculate unit scale vector
        current_model = [param.data().copy() for param in net.collect_params().values()]
        history_norm = nd.norm(history)
        last_grad_norm = nd.norm(last_grad)
        scale = nd.norm(history - nd.expand_dims(last_grad, axis=-1)* history_norm/(last_grad_norm+1e-9), axis=1)
        deviation = scale * fixed_rand / (nd.norm(scale)+1e-9)
        
        # calculate scaling factor lambda
        if e % 50 == 0:
            total_update = nd.concat(*[xx.reshape((-1, 1)) for xx in current_model],
                                dim=0) - nd.concat(*[xx.reshape((-1, 1)) for xx in last_50_model], dim=0)
            total_update = nd.where(total_update == 0, nd.concat(*[xx.reshape((-1, 1)) for xx in current_model],dim=0), total_update)
            current_sign = nd.sign(total_update)
            aligned_dim_cnt = (current_sign == nd.expand_dims(fixed_rand, axis=-1)).sum() 
            if aligned_dim_cnt < k_99 and scaling_factor*0.7>=0.5:
                sf = scaling_factor*0.7
            else:
                sf = scaling_factor
            lamda_succ = sf * history_norm
        else:
            sf = scaling_factor
            lamda_succ = sf * history_norm
        mal_update = lamda_succ * deviation
        for i in range(nfake):
            v[i] = nd.expand_dims(mal_update, axis=-1)
    return v, sf








def random_attack(v, net, lr, nfake, history, fixed_rand,  init_model, last_50_model, last_grad,e, scaling_factor=100000.):
    for i in range(nfake):
        v[i] = scaling_factor * nd.random.normal(loc=0, scale=1, shape=v[0].shape)
    return v, scaling_factor


def init_attack(v, net, lr, nfake, history, fixed_rand, init_model, last_50_model, last_grad,e, scaling_factor=100000.):
    current_model = [param.data().copy() for param in net.collect_params().values()]
    direction = nd.concat(*[xx.reshape((-1, 1)) for xx in init_model], dim=0) - nd.concat(*[xx.reshape((-1, 1)) for xx in current_model], dim=0)
    for i in range(nfake):
        v[i] = scaling_factor * direction
    return v, scaling_factor

