
import torch
import numpy as np
import json
from collections import defaultdict
import os


def cal_cos(grad_1, grad_2):
    """Calculate cosine similarity between two gradients"""
    return torch.dot(grad_1.flatten(), grad_2.flatten()) / (torch.norm(grad_1) + 1e-9) / (torch.norm(grad_2) + 1e-9)


def median_grad(gradients):
    """Compute median of gradients
    
    Args:
        gradients: list of gradient lists, where each gradient list contains tensors for each parameter
    
    Returns:
        median gradient as a 1D tensor
    """
    # Flatten and concatenate all parameters for each client
    param_list = [torch.cat([xx.reshape(-1, 1) for xx in x], dim=0) for x in gradients]
    
    # Concatenate all clients' gradients
    sorted_arr, _ = torch.sort(torch.cat(param_list, dim=1), dim=-1)
    median_idx = sorted_arr.shape[-1] // 2
    
    if sorted_arr.shape[-1] % 2 == 0:
        # if the number of elements is even, take the average of the two middle elements
        median = torch.mean(sorted_arr[:, median_idx-1:median_idx+1], dim=-1)
    else:
        # if the number of elements is odd, take the middle element
        median = sorted_arr[:, median_idx]

    return median


def read_dir(data_dir):
    clients = []
    groups = []
    data = defaultdict(lambda: None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return clients, groups, data


def read_data(train_data_dir, test_data_dir):
    """Parses data in given train and test data directories
    
    Assumes:
    - the data in the input directories are .json files with
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Args:
        train_data_dir: directory containing training data JSON files
        test_data_dir: directory containing test data JSON files

    Returns:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    """
    train_clients, train_groups, train_data = read_dir(train_data_dir)
    test_clients, test_groups, test_data = read_dir(test_data_dir)

    assert train_clients == test_clients
    assert train_groups == test_groups

    return train_clients, train_groups, train_data, test_data


def flatten_gradients(gradients):
    """Flatten a list of gradient tensors into a single 1D tensor
    
    Args:
        gradients: list of tensors (gradients for each parameter)
    
    Returns:
        flattened 1D tensor
    """
    return torch.cat([g.flatten() for g in gradients])


def unflatten_gradients(flat_grad, shapes):
    """Unflatten a 1D gradient tensor back into a list of tensors with given shapes
    
    Args:
        flat_grad: 1D flattened gradient tensor
        shapes: list of shapes for each parameter
    
    Returns:
        list of gradient tensors with original shapes
    """
    gradients = []
    idx = 0
    for shape in shapes:
        numel = int(np.prod(shape))
        gradients.append(flat_grad[idx:idx+numel].reshape(shape))
        idx += numel
    return gradients


def compute_gradient_norm(gradients):
    """Compute L2 norm of gradients
    
    Args:
        gradients: list of gradient tensors
    
    Returns:
        L2 norm as a scalar
    """
    return torch.norm(flatten_gradients(gradients))


def aggregate_gradients(gradient_list, aggregation_fn):
    """Apply an aggregation function to a list of gradients
    
    Args:
        gradient_list: list of gradient lists (one per client)
        aggregation_fn: function to aggregate (e.g., torch.mean, torch.median)
    
    Returns:
        aggregated gradient list
    """
    # Stack all gradients
    num_params = len(gradient_list[0])
    aggregated = []
    
    for param_idx in range(num_params):
        param_grads = torch.stack([g[param_idx] for g in gradient_list])
        aggregated.append(aggregation_fn(param_grads, dim=0))
    
    return aggregated
