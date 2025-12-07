from __future__ import print_function
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import nd_aggregation_pytorch as nd_aggregation
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import random
import argparse
import byzantine_pytorch as byzantine
import seaborn as sns
import matplotlib.pyplot as plt
import os
from torchsummary import summary

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from utils_pytorch import *
import wandb
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_pc", help="the number of data the server holds", type=int, default=100)
    parser.add_argument("--dataset", help="dataset", type=str, default="FashionMNIST")
    parser.add_argument("--bias", help="degree of non-iid", type=float, default=0.5)
    parser.add_argument("--net", help="net", type=str, default="cnn")
    parser.add_argument("--batch_size", help="batch size", type=int, default=32)
    parser.add_argument("--lr", help="learning rate", type=float, default=0.01)
    parser.add_argument("--nworkers", help="# workers", type=int, default=1200)
    parser.add_argument("--niter", help="# iterations", type=int, default=200)
    parser.add_argument("--gpu", help="index of gpu", type=int, default=0)
    parser.add_argument("--seed", help="seed", type=int, default=42)
    parser.add_argument("--selected_layer", help="selected_layer", type=int, default=0)
    parser.add_argument("--nfake", help="# fake clients", type=int, default=100)
    parser.add_argument("--byz_type", help="type of attack", type=str, default="no")
    parser.add_argument("--aggregation", help="aggregation", type=str, default="fltrust")
    parser.add_argument("--p", help="bias probability of 1 in server sample", type=float, default=0.1)
    parser.add_argument("--sf", help="scaling factor", type=float, default=10)
    parser.add_argument("--participation_rate",help="participation_rate", type=float, default=0.025)
    # parser.add_argument("--participation_rate",help="participation_rate", type=float, default=0.1)
    # parser.add_argument("--step", help="period to log accuracy", type=int, default=100)
    parser.add_argument("--step", help="period to log accuracy", type=int, default=20)
    parser.add_argument("--local_epoch", help="local_epoch", type=int, default=0)

    return parser.parse_args()

def get_device(device):
    # define the device to use
    print('device', device  )
    if device == -1:
        return torch.device('cpu')
    else:
        return torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')
    
def get_dnn(num_outputs=600, input_dim=600):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_dim, 1024),
        nn.Tanh(),
        nn.Linear(1024, num_outputs)
    )

class CNN(nn.Module):
    def __init__(self, num_outputs=10, input_channels=1, input_size=28):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 30, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(30, 50, kernel_size=3)
        
        # Calculate the size after convolutions and pooling
        # After conv1 (k=3): size - 2
        # After pool1: (size - 2) // 2
        # After conv2 (k=3): ((size - 2) // 2) - 2
        # After pool2: (((size - 2) // 2) - 2) // 2
        conv_output_size = (((input_size - 2) // 2) - 2) // 2
        self.fc_input_size = 50 * conv_output_size * conv_output_size
        
        self.fc1 = nn.Linear(self.fc_input_size, 100)
        self.fc2 = nn.Linear(100, num_outputs)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.fc_input_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_cnn(num_outputs=10, input_channels=1, input_size=28):
    # define the architecture of the CNN
    return CNN(num_outputs, input_channels, input_size)


class CNNCifar(nn.Module):
    def __init__(self, num_outputs=10, input_channels=3, input_size=32):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        
        # Calculate the size after convolutions and pooling
        conv_output_size = (((input_size - 2) // 2) - 2) // 2
        self.fc_input_size = 64 * conv_output_size * conv_output_size
        
        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.fc2 = nn.Linear(512, num_outputs)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.fc_input_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_cnn_cifar(num_outputs=10, input_channels=3, input_size=32):
    return CNNCifar(num_outputs, input_channels, input_size)



def get_net(net_type, num_outputs=10, num_inputs=(1, 1, 28, 28)):
    # define the model architecture
    # num_inputs format: (batch, channels, height, width) or (batch, features)
    if net_type == 'cnn':
        input_channels = num_inputs[1]
        input_size = num_inputs[2]  # Assume square images
        net = get_cnn(num_outputs, input_channels, input_size)
    elif net_type == "cnn_cifar":
        input_channels = num_inputs[1]
        input_size = num_inputs[2]
        net = get_cnn_cifar(num_outputs, input_channels, input_size)
    elif net_type == 'dnn':
        # For DNN, flatten all dimensions except batch
        input_dim = int(np.prod(num_inputs[1:]))
        net = get_dnn(num_outputs, input_dim)
    else:
        raise NotImplementedError
    return net


def get_shapes(dataset):
    # determine the input/output shapes 
    if dataset == 'FashionMNIST' or dataset == 'mnist':
        num_inputs = (1, 1, 28, 28)
        num_outputs = 10
        num_labels = 10
    elif dataset == 'FEMNIST':
        num_inputs = (1, 1, 28, 28)
        num_outputs = 62
        num_labels = 62
    elif dataset == 'cifar10':
        num_inputs = (1, 3, 32, 32)
        num_outputs = 10
        num_labels = 10
    elif args.dataset == 'purchase':
        num_inputs = (1, 600)
        num_outputs = 100
        num_labels = 100
    else:
        raise NotImplementedError
    return num_inputs, num_outputs, num_labels

def evaluate_accuracy(data_iterator, net, device):
    # evaluate the (attack) accuracy of the model
    net.eval()
    correct = 0
    total = 0
    # with torch.no_grad():
    with torch.inference_mode():
        for data, label in data_iterator:
            data = data.to(device)
            label = label.to(device).long()
            output = net(data)
            predictions = torch.argmax(output, dim=1)
            correct += (predictions == label).sum().item()
            total += label.size(0)
    net.train()
    return correct / total if total > 0 else 0


def get_byz(byz_type):
    # get the attack type
    if byz_type == "no":
        return byzantine.no_byz
    elif byz_type == 'trim_attack':
        return byzantine.fang_attack
    elif byz_type == 'lie_attack':
        return byzantine.lie_attack
    elif byz_type == 'dyn_attack':
        return byzantine.opt_fang
    elif byz_type == 'min_max':
        return byzantine.min_max
    elif byz_type == 'min_sum':
        return byzantine.min_sum
    elif byz_type == 'init_attack':
        return byzantine.init_attack
    elif byz_type == 'random_attack':
        return byzantine.random_attack
    elif byz_type == "poisonedfl":
        return byzantine.poisonedfl
    else:
        raise NotImplementedError
        
def load_data(dataset):
    # load the dataset
    from torchvision import datasets, transforms
    if dataset == 'FashionMNIST':
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
        train_data = DataLoader(train_dataset, batch_size=60000, shuffle=True)
        test_data = DataLoader(test_dataset, batch_size=250, shuffle=False)
    elif dataset == 'mnist':
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
        train_data = DataLoader(train_dataset, batch_size=60000, shuffle=True)
        test_data = DataLoader(test_dataset, batch_size=250, shuffle=False)
    elif dataset == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
        train_data = DataLoader(train_dataset, batch_size=128, shuffle=True)
        test_data = DataLoader(test_dataset, batch_size=128, shuffle=False)
    elif args.dataset == 'purchase':
        cache_dir = './.cache'
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            print(f"Created cache directory: {cache_dir}")
        data_cache_path = os.path.join(cache_dir, "purchase_data.npy")
        if os.path.exists(data_cache_path):
            print(f"Loading Purchase dataset from cache: {data_cache_path} ...")
            all_data = np.load(data_cache_path)
            print(f"Loaded Purchase dataset with {all_data.shape[0]} samples from cache.")
        else: 
            print("Parsing Purchase dataset from CSV (First time run)...")
            all_data = np.genfromtxt("./purchase/dataset_purchase", skip_header=1, delimiter=',')
            print(f"Saving Purchase dataset cache to {data_cache_path} ...")
            np.save(data_cache_path, all_data)
            print(f"Saved Purchase dataset cache to {data_cache_path}.")
        shuffle_index = np.random.permutation(all_data.shape[0])
        all_data = all_data[shuffle_index]
        each_worker_data = [torch.tensor(all_data[150*i:150*(i+1), 1:] * 2. - 1, dtype=torch.float32) for i in range(1200)]
        each_worker_label = [torch.tensor(all_data[150*i:150*(i+1), 0] - 1, dtype=torch.long) for i in range(1200)]   
        train_data = (each_worker_data, each_worker_label)
        test_data = ((torch.tensor(all_data[180000:, 1:] * 2. - 1, dtype=torch.float32), torch.tensor(all_data[180000:, 0] - 1, dtype=torch.long)),)
    elif args.dataset == "FEMNIST":
        cache_dir = './.cache'
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            print(f"Created cache directory: {cache_dir}")
        
        train_cache_path = os.path.join(cache_dir, "femnist_train.pt")
        test_cache_path = os.path.join(cache_dir, "femnist_test.pt")
        
        if os.path.exists(train_cache_path):
            print(f"Loading FEMNIST from cache: {train_cache_path} ...")
            # 直接載入處理好的 Tensor 列表，速度極快
            cached_data = torch.load(train_cache_path)
            each_worker_data = cached_data['x']
            each_worker_label = cached_data['y']
            each_worker_num = cached_data['n']
            print(f"Loaded {len(each_worker_data)} workers from cache.")
        else:
            print("Parsing FEMNIST from JSON (First time run)...")
            import json
            each_worker_data = []
            each_worker_label = []
            each_worker_num = []
            for i in range(30):
                filestring = "./leaf/data/femnist/data/train/" + \
                    "all_data_"+str(i) + "_niid_0_keep_100_train_9.json"
                with open(filestring, 'r') as f:
                    load_dict = json.load(f)
                    each_worker_num.extend(load_dict['num_samples'])
                    for user in load_dict['users']:
                        x = torch.tensor(load_dict['user_data'][user]['x'], dtype=torch.float32).reshape(-1, 1, 28, 28)
                        y = torch.tensor(load_dict['user_data'][user]['y'], dtype=torch.long)

                        each_worker_data.append(x)
                        each_worker_label.append(y)
            print(f"Saving cache to {train_cache_path}...")
            torch.save({
                'x': each_worker_data,
                'y': each_worker_label,
                'n': each_worker_num
            }, train_cache_path)

        # random shuffle the workers
        random_order = np.random.RandomState(
            seed=args.seed).permutation(args.nworkers)
        each_worker_data = [each_worker_data[i] for i in random_order]
        each_worker_label = [each_worker_label[i] for i in random_order]
        each_worker_num = torch.tensor([each_worker_num[i]
                                   for i in random_order])
        train_data = (each_worker_data, each_worker_label)

        if os.path.exists(test_cache_path):
            print(f"Loading FEMNIST Test data from cache: {test_cache_path} ...")
            cached_data = torch.load(test_cache_path)
            test_x = cached_data['x']
            test_y = cached_data['y']
            print(f"Loaded test data with {test_x.shape[0]} samples from cache.")
        else:
            print("Parsing FEMNIST Test data from JSON (First time run)...")
            train_data_dir = os.path.join(
                "./leaf/data/femnist/data", "train")
            test_data_dir = os.path.join(
                "./leaf/data/femnist/data", "test")
            data = read_data(train_data_dir, test_data_dir)
            users, groups, train_data_ori, test_data_ori = data
            test_x = torch.cat([torch.tensor(test_data_ori[u]['x'], dtype=torch.float32).reshape(-1,1, 28, 28) for u in users], dim=0)
            test_y = torch.cat([torch.tensor(test_data_ori[u]['y'], dtype=torch.long) for u in users], dim=0)
            print(f"Saving test cache to {test_cache_path}...")
            torch.save({
                'x': test_x,
                'y': test_y
            }, test_cache_path)

        test_dataset = TensorDataset(test_x, test_y)
        test_data = DataLoader(test_dataset, batch_size=250, shuffle=False)
    else: 
        raise NotImplementedError
    return train_data, test_data
    

def assign_data(train_data, bias, device, num_labels=10, num_workers=100, server_pc=100, p=0.1, dataset="FashionMNIST", seed=1, num_inputs=(1, 561)):
    if dataset == "purchase":
        server_data = []
        server_label = [] 
        for i in range(len(train_data[0])):
            if i >= server_pc:
                break
            rd = random.randint(1, train_data[0][i].shape[0]-1)
            server_data.append(train_data[0][i][rd].unsqueeze(0))
            server_label.append(train_data[1][i][rd].unsqueeze(0))
        server_data = torch.cat(server_data, dim=0) if server_pc > 0 else None
        server_label = torch.cat(server_label, dim=0) if server_pc > 0 else None
        return server_data, server_label, train_data[0], train_data[1]
    
    elif dataset == "FEMNIST":
        print(len(train_data[0]), train_data[0][0].shape)
        server_data = []
        server_label = [] 
        for i in range(len(train_data[0])):
            if i >= server_pc:
                break
            rd = random.randint(1, train_data[0][i].shape[0]-1)
            server_data.append(train_data[0][i][rd].unsqueeze(0)) ####
            server_label.append(train_data[1][i][rd].unsqueeze(0))
        server_data = torch.cat(server_data, dim=0) if server_pc > 0 else None
        server_label = torch.cat(server_label, dim=0) if server_pc > 0 else None
        return server_data, server_label, train_data[0], train_data[1]

    elif dataset == "FashionMNIST" or dataset == "mnist":
        # assign data to the clients
        other_group_size = (1 - bias) / (num_labels - 1)
        worker_per_group = num_workers / num_labels

        #assign training data to each worker
        each_worker_data = [[] for _ in range(num_workers)]
        each_worker_label = [[] for _ in range(num_workers)]   
        server_data = []
        server_label = [] 
        
        # compute the labels needed for each class
        real_dis = [1. / num_labels for _ in range(num_labels)]
        samp_dis = [0 for _ in range(num_labels)]
        num1 = int(server_pc * p)
        samp_dis[1] = num1
        average_num = (server_pc - num1) / (num_labels - 1)
        resid = average_num - np.floor(average_num)
        sum_res = 0.
        for other_num in range(num_labels - 1):
            if other_num == 1:
                continue
            samp_dis[other_num] = int(average_num)
            sum_res += resid
            if sum_res >= 1.0:
                samp_dis[other_num] += 1
                sum_res -= 1
        samp_dis[num_labels - 1] = server_pc - np.sum(samp_dis[:num_labels - 1])

        # randomly assign the data points based on the labels
        server_counter = [0 for _ in range(num_labels)]
        for _, (data, label) in enumerate(train_data):
            for (x, y) in zip(data, label):
                x = x.to(device).reshape(1,1,28,28)
                y = y.to(device).view(-1)
                
                upper_bound = (y.item()) * (1. - bias) / (num_labels - 1) + bias
                lower_bound = (y.item()) * (1. - bias) / (num_labels - 1)
                rd = np.random.random_sample()
                
                if rd > upper_bound:
                    worker_group = int(np.floor((rd - upper_bound) / other_group_size) + y.item() + 1)
                elif rd < lower_bound:
                    worker_group = int(np.floor(rd / other_group_size))
                else:
                    worker_group = y.item()
                
                if server_counter[int(y.item())] < samp_dis[int(y.item())]:
                    server_data.append(x)
                    server_label.append(y)
                    server_counter[int(y.item())] += 1
                else:
                    rd = np.random.random_sample()
                    selected_worker = int(worker_group * worker_per_group + int(np.floor(rd * worker_per_group)))
                    each_worker_data[selected_worker].append(x)
                    each_worker_label[selected_worker].append(y)
                    
        server_data = torch.cat(server_data, dim=0) if server_pc > 0 else None
        server_label = torch.cat(server_label, dim=0) if server_pc > 0 else None
        
        each_worker_data = [torch.cat(each_worker, dim=0) for each_worker in each_worker_data] 
        each_worker_label = [torch.cat(each_worker, dim=0) for each_worker in each_worker_label]        
        random_order = np.random.RandomState(seed=seed).permutation(num_workers)
        each_worker_data = [each_worker_data[i] for i in random_order]
        each_worker_label = [each_worker_label[i] for i in random_order]
        return server_data, server_label, each_worker_data, each_worker_label
    
    elif dataset == "cifar10":
        # assign data to the clients
        other_group_size = (1 - bias) / (num_labels - 1)
        worker_per_group = num_workers / num_labels

        #assign training data to each worker
        each_worker_data = [[] for _ in range(num_workers)]
        each_worker_label = [[] for _ in range(num_workers)]
        server_data = []
        server_label = []

        # compute the labels needed for each class
        real_dis = [1. / num_labels for _ in range(num_labels)]
        samp_dis = [0 for _ in range(num_labels)]
        num1 = int(server_pc * p)
        samp_dis[1] = num1
        average_num = (server_pc - num1) / (num_labels - 1)
        resid = average_num - np.floor(average_num)
        sum_res = 0.
        for other_num in range(num_labels - 1):
            if other_num == 1:
                continue
            samp_dis[other_num] = int(average_num)
            sum_res += resid
            if sum_res >= 1.0:
                samp_dis[other_num] += 1
                sum_res -= 1
        samp_dis[num_labels - 1] = server_pc - \
            np.sum(samp_dis[:num_labels - 1])

        # randomly assign the data points based on the labels
        server_counter = [0 for _ in range(num_labels)]
        for _, (data, label) in enumerate(train_data):
            for (x, y) in zip(data, label):
                x = x.to(device).reshape(1, 3, 32, 32)
                y = y.to(device).view(-1)

                upper_bound = (y.item()) * (1. - bias) / \
                    (num_labels - 1) + bias
                lower_bound = (y.item()) * (1. - bias) / (num_labels - 1)
                rd = np.random.random_sample()

                if rd > upper_bound:
                    worker_group = int(
                        np.floor((rd - upper_bound) / other_group_size) + y.item() + 1)
                elif rd < lower_bound:
                    worker_group = int(np.floor(rd / other_group_size))
                else:
                    worker_group = y.item()

                if server_counter[int(y.item())] < samp_dis[int(y.item())]:
                    server_data.append(x)
                    server_label.append(y)
                    server_counter[int(y.item())] += 1
                else:
                    rd = np.random.random_sample()
                    selected_worker = int(
                        worker_group * worker_per_group + int(np.floor(rd * worker_per_group)))
                    each_worker_data[selected_worker].append(x)
                    each_worker_label[selected_worker].append(y)

        server_data = torch.cat(server_data, dim=0) if server_pc > 0 else None
        server_label = torch.cat(
            server_label, dim=0) if server_pc > 0 else None

        each_worker_data = [torch.cat(each_worker, dim=0)
                            for each_worker in each_worker_data]
        each_worker_label = [torch.cat(each_worker, dim=0)
                             for each_worker in each_worker_label]

        # randomly permute the workers
        random_order = np.random.RandomState(
            seed=seed).permutation(num_workers)
        each_worker_data = [each_worker_data[i] for i in random_order]
        each_worker_label = [each_worker_label[i] for i in random_order]

        return server_data, server_label, each_worker_data, each_worker_label


def select_clients(clients, frac=1.0):
    if frac != 1:
        return random.sample(clients, int(len(clients)*frac)) 
    else:
        return clients
        

def main(args):

    # Initialize wandb
    wandb.init(
        # project name = PoisonedFL_{attack type}_{aggregation}
        project=f"pytorch_PoisonedFL",
        #project="PoisonedFL",
        config={
            "dataset": args.dataset,
            "net": args.net,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "nworkers": args.nworkers,
            "niter": args.niter,
            "nfake": args.nfake,
            "byz_type": args.byz_type,
            "aggregation": args.aggregation,
            "bias": args.bias,
            "sf": args.sf,
            "participation_rate": args.participation_rate,
            "local_epoch": args.local_epoch,
            "seed": args.seed,
            "server_pc": args.server_pc,
            "p": args.p,
        }
    )

    device = get_device(args.gpu)
    print('device', device)

    # set parameters
    batch_size = args.batch_size
    num_inputs, num_outputs, num_labels = get_shapes(args.dataset)
    byz = get_byz(args.byz_type)
    num_workers = args.nworkers
    lr = args.lr
    niter = args.niter

    net = get_net(args.net, num_outputs, num_inputs)
    net = net.to(device)

    # Initialize parameters
    def init_weights(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=2.24)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    net.apply(init_weights)
    
    # loss
    criterion = nn.CrossEntropyLoss(reduction='sum')

    grad_list = []
    test_acc_list = []

    # load the data
    seed = args.seed
    if seed > 0:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
    print("Loading data...")
    train_data, test_data = load_data(args.dataset)
    
    # assign data to the server and clients
    server_data, server_label, each_worker_data, each_worker_label = assign_data(
                                                                train_data, args.bias, device, num_labels=num_labels, num_workers=num_workers, 
                                                                server_pc=args.server_pc, p=args.p, dataset=args.dataset, seed=seed,num_inputs=num_inputs)
    # server_data can be considered as the eval data (choosing from server_pc clients each having one data point)
    print("server data shape:", server_data.shape, server_label.shape)
    print(each_worker_data[0].shape, each_worker_label[0].shape)
    # run a forward pass to really initialize the model
    data_count = []
    for data in each_worker_data:
        data_count.append(data.shape[0])
    with torch.no_grad():
        net(torch.zeros(num_inputs, device=device))
    print(torch.zeros(num_inputs, device=device).shape)
    # summary(net, num_inputs[1:])
    # set initial parameters
    init_model = [param.data.clone() for param in net.parameters()]
    last_model = [param.data.clone() for param in net.parameters()]
    history = None
    last_50_model = None
    last_grad = None
    sf = args.sf

    # set the fixed s vector for poisonedfl
    fixed_rand = torch.sign(torch.randn(
        torch.cat([xx.reshape(-1, 1) for xx in init_model], dim=0).shape, device=device)).squeeze()

    avg_loss = 0

    # begin training     
    V_ref = None
    server_grads = None
    net.train()
    # for e in tqdm(range(niter)):    
    for e in range(niter):   
        participating_clients = select_clients(
            range(num_workers) , args.participation_rate)
        
        # caculate the number of fake clients
        probability = args.nfake * args.participation_rate - int(args.nfake * args.participation_rate)
        if random.random() >= probability:
            parti_nfake = int(args.nfake * args.participation_rate)
        else:
            parti_nfake = int(args.nfake * args.participation_rate) + 1
        
        # gradients for fake clients
        for i in range(parti_nfake):
            grad_list.append([torch.zeros_like(param.data) for param in net.parameters()])
            
        print("start processing clients of iteration %d" % e)
        # gradients for genuine clients
        for i in participating_clients:
            ori_para = [param.data.clone() for param in net.parameters()]
            for _ in range(args.local_epoch):
                shuffled_order = np.random.choice(list(range(each_worker_data[i].shape[0])), size=each_worker_data[i].shape[0], replace=False)
                for b_id in range(max(each_worker_data[i].shape[0]//batch_size, 1)):
                    if batch_size >= each_worker_data[i].shape[0]:
                        minibatch = list(range(each_worker_data[i].shape[0]))
                    else:
                        minibatch = shuffled_order[b_id * batch_size: (b_id +1) * batch_size]
                    
                    # Zero gradients
                    for param in net.parameters():
                        param.grad = None
                    
                    output = net(each_worker_data[i][minibatch].to(device))
                    loss = criterion(output, each_worker_label[i][minibatch].to(device).long())
                    loss.backward()
                    avg_loss += loss.item()
                    
                    with torch.no_grad():
                        # nn.utils.clip_grad_norm_(net.parameters(), max_norm=10.0)
                        for param in net.parameters():
                            if param.grad is not None:
                                param.data.sub_(lr/batch_size * param.grad.data)
                    
            grad_list.append([( param.data.clone()- ori_data.clone()) for param, ori_data in zip(net.parameters(), ori_para)])
            with torch.no_grad():
                for param, ori_data in zip(net.parameters(), ori_para):
                    param.data.copy_(ori_data)
        
        print("finished processing clients of iteration %d" % e)
            
        # print('grad_shape', len(grad_list[-1]),len(grad_list[0]))
        if args.aggregation == "specguard" and (V_ref is None or e % 20 == 0):
            # need to compute the gradients for each server data and compute compact.SVD
            server_grads = []
            params = list(net.parameters())
            for i in range(server_data.shape[0]):
                # simply use net to compute the gradients

                ## zero the gradients
                for param in params:
                    param.grad = None
                
                output = net(server_data[i].reshape((1,)+num_inputs[1:]).to(device))
                loss = criterion(output, server_label[i].reshape((1,)).to(device).long())
                loss.backward()
                # nn.utils.clip_grad_norm_(net.parameters(), max_norm=10.0)
                server_grads.append(torch.cat([param.grad.reshape(-1, 1) for param in params if param.grad is not None], dim=0).reshape(1, -1))
            server_grads = torch.cat(server_grads, dim=0).reshape((server_data.shape[0], -1))

            V_ref_base_k = 5
            
            ## center the gradients
            mean_grad = torch.mean(server_grads, dim=0, keepdim=True)
            centered_grads = server_grads - mean_grad
            
            ## SVD (GPU-accelerated)
            # print(centered_grads.shape)
            U, S, Vt = torch.linalg.svd(centered_grads, full_matrices=False)
            V_ref = Vt[:V_ref_base_k, :] #(k,d)

        elif args.aggregation == "fltrust" and (server_grads is None or e % 20 == 0):
            ori_para = [param.data.clone() for param in net.parameters()]
            for _ in range(args.local_epoch):
                shuffled_order = np.random.choice(list(range(server_data.shape[0])), size=server_data.shape[0], replace=False)
                for b_id in range(max(server_data.shape[0]//batch_size, 1)):
                    if batch_size >= server_data.shape[0]:
                        minibatch = list(range(server_data.shape[0]))
                    else:
                        minibatch = shuffled_order[b_id * batch_size: (b_id +1) * batch_size]
                    
                    # Zero gradients
                    for param in net.parameters():
                        param.grad = None
                    
                    output = net(server_data[minibatch].to(device))
                    loss = criterion(output, server_label[minibatch].to(device).long())
                    loss.backward()
                    avg_loss += loss.item()
                    
                    with torch.no_grad():
                        # nn.utils.clip_grad_norm_(net.parameters(), max_norm=10.0)
                        for param in net.parameters():
                            if param.grad is not None:
                                param.data.sub_(lr/batch_size * param.grad.data)
                    
            server_grads = [( param.data.clone()- ori_data.clone()) for param, ori_data in zip(net.parameters(), ori_para)]
            with torch.no_grad():
                for param, ori_data in zip(net.parameters(), ori_para):
                    param.data.copy_(ori_data)

        try:
            avg_loss = avg_loss/len(participating_clients)
            print("Iteration %02d. Avg_loss %0.4f" % (e, avg_loss))
            # Log training loss to wandb
            wandb.log({"train/loss": avg_loss, "iteration": e})
        except:
            import pdb
            pdb.set_trace()

        avg_loss = 0
        if not grad_list:
            continue
        if args.aggregation == "mean":
            return_pare_list, sf = nd_aggregation.simple_mean(
            grad_list, net, lr / batch_size, parti_nfake, byz, history, fixed_rand, init_model, last_50_model, last_grad, sf, e)    
        elif args.aggregation == "trim":
            return_pare_list, sf = nd_aggregation.trim(
                grad_list, net, lr / batch_size, parti_nfake, byz, history, fixed_rand, init_model, last_50_model, last_grad, sf, e)
        elif args.aggregation == "median":
            return_pare_list, sf = nd_aggregation.median(
                grad_list, net, lr / batch_size, parti_nfake, byz, history, fixed_rand, init_model, last_50_model, last_grad, sf, e)
        elif args.aggregation == "mean_norm":
            return_pare_list, sf = nd_aggregation.mean_norm(
                grad_list, net, lr / batch_size, parti_nfake, byz, history,fixed_rand, init_model, last_50_model, last_grad, sf, e)
        elif args.aggregation == "specguard":
            return_pare_list, sf, retained_count = nd_aggregation.specguard(
                grad_list, net, lr / batch_size, parti_nfake, byz, history,fixed_rand, init_model, last_50_model, last_grad, sf, e, V_ref)
        elif args.aggregation == "fltrust":
            return_pare_list, sf = nd_aggregation.fltrust(
                grad_list, net, lr / batch_size, parti_nfake, byz, history,fixed_rand, init_model, last_50_model, last_grad, sf, e, server_grads)
        elif args.aggregation == "no":
            return_pare_list, sf = nd_aggregation.no_aggregation(
                grad_list, net, lr / batch_size, parti_nfake, byz, history,fixed_rand, init_model, last_50_model, last_grad, sf, e)
            #raise NotImplementedError
        if parti_nfake != 0: # last_grad: the previous mean malicious update
            if "norm" in args.aggregation:
                last_grad = torch.mean(return_pare_list[:,:parti_nfake], dim=-1).clone()
            else:
                last_grad = torch.mean(
                    torch.cat([return_pare_list[i] for i in range(parti_nfake)], dim=1), dim=-1).clone()
            # print("last_grad shape:", last_grad.shape,last_grad)
        del grad_list
        del return_pare_list
        grad_list = []
        current_model = [param.data.clone() for param in net.parameters()]
        if (e + 1) % args.step == 0 or e + 20 >= args.niter:
            test_accuracy = evaluate_accuracy(test_data, net, device)
            test_acc_list.append(test_accuracy)
            print("Iteration %02d. Test_acc %0.4f" % (e, test_accuracy))
            # Log test accuracy to wandb
            wandb.log({"test/accuracy": test_accuracy, "iteration": e})
            
        if e % 50 == 0:
            last_50_model = current_model 

        # history: last global update
        history = (torch.cat([xx.reshape(-1, 1) for xx in current_model], dim=0) - torch.cat([xx.reshape(-1, 1) for xx in last_model], dim=0) )
        last_model = [param.data.clone() for param in net.parameters()]
        
        # Log scaling factor for poisonedfl attack
        if args.byz_type == "poisonedfl":
            wandb.log({"attack/scaling_factor": sf, "iteration": e})
            if args.aggregation == "specguard":
                wandb.log({"defense/retained_count": retained_count, "iteration": e})

        from os import path
    
    # Log final test accuracy
    final_test_acc = evaluate_accuracy(test_data, net, device)
    wandb.log({"test/final_accuracy": final_test_acc})
    print("Final Test Accuracy: %0.4f" % final_test_acc)
    
    # Finish wandb run
    wandb.finish()
            
    del test_acc_list
    test_acc_list = []

   
if __name__ == "__main__":
    args = parse_args()
    main(args)
