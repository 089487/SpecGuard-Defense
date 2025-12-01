from __future__ import print_function
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import nd_aggregation
import numpy as np
np.bool = np.bool_
import mxnet as mx
from mxnet import nd, autograd, gluon
import random
import argparse
import byzantine
import seaborn as sns
import matplotlib.pyplot as plt
import os

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from utils import *
import torch
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
    parser.add_argument("--step", help="period to log accuracy", type=int, default=1000)
    parser.add_argument("--local_epoch", help="local_epoch", type=int, default=0)

    return parser.parse_args()

def get_device(device):
    # define the device to use
    if device == -1:
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(device)
    return ctx
    
def get_dnn(num_outputs=600):
    dnn = gluon.nn.Sequential()
    with dnn.name_scope():
        dnn.add(gluon.nn.Dense(1024, activation='tanh'))
        dnn.add(gluon.nn.Dense(num_outputs))
    return dnn

def get_cnn(num_outputs=10):
    # define the architecture of the CNN
    cnn = gluon.nn.Sequential()
    with cnn.name_scope():
        cnn.add(gluon.nn.Conv2D(channels=30, kernel_size=3, activation='relu'))
        cnn.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        cnn.add(gluon.nn.Conv2D(channels=50, kernel_size=3, activation='relu'))
        cnn.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        cnn.add(gluon.nn.Flatten())
        cnn.add(gluon.nn.Dense(100, activation="relu"))
        cnn.add(gluon.nn.Dense(num_outputs))
    return cnn


def get_cnn_cifar(num_outputs=10):
    cnn = gluon.nn.Sequential()
    with cnn.name_scope():
        cnn.add(gluon.nn.Conv2D(channels=32, kernel_size=3,in_channels=3, activation='relu'))
        cnn.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        cnn.add(gluon.nn.Conv2D(channels=64, kernel_size=3, activation='relu'))
        cnn.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        cnn.add(gluon.nn.Flatten())
        cnn.add(gluon.nn.Dense(512, activation="relu"))
        cnn.add(gluon.nn.Dense(num_outputs))
    return cnn



def get_net(net_type, num_outputs=10):
    # define the model architecture
    if net_type == 'cnn':
        net = get_cnn(num_outputs)
    elif net_type == "cnn_cifar":
        net = get_cnn_cifar(num_outputs)
    elif net_type == 'dnn':
        net = get_dnn(num_outputs)
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

def evaluate_accuracy(data_iterator, net, ctx):
    # evaluate the (attack) accuracy of the model
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        remaining_idx = list(range(data.shape[0]))
        output = net(data)
        predictions = nd.argmax(output, axis=1)                
        predictions = predictions[remaining_idx]
        label = label[remaining_idx]
        acc.update(preds=predictions, labels=label)        
    return acc.get()[1]


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
    if dataset == 'FashionMNIST':
        def transform(data, label):
            return nd.transpose(data.astype(np.float32), (2, 0, 1)) / 255, label.astype(np.float32)
        train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.FashionMNIST(train=True, transform=transform), 60000,shuffle=True, last_batch='rollover')
        test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.FashionMNIST(train=False, transform=transform), 250, shuffle=False, last_batch='rollover')
    elif dataset == 'mnist':
        def transform(data, label):
            return nd.transpose(data.astype(np.float32), (2, 0, 1)) / 255, label.astype(np.float32)
        train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=True, transform=transform), 60000,shuffle=True, last_batch='rollover')
        test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform), 250, shuffle=False, last_batch='rollover')
    elif dataset == 'cifar10':
        def transform(data, label):
            data = nd.transpose(data.astype(np.float32), (2, 0, 1)) / 255
            return data, label.astype(np.float32)
        train_data = mx.gluon.data.DataLoader(
            mx.gluon.data.vision.CIFAR10(train=True, transform=transform),
            batch_size=128,
            shuffle=True,
            last_batch='rollover')
        test_data = mx.gluon.data.DataLoader(
            mx.gluon.data.vision.CIFAR10(train=False, transform=transform),
            batch_size=128,
            shuffle=False,
            last_batch='rollover')
    elif args.dataset == 'purchase':
        all_data = np.genfromtxt("./purchase/dataset_purchase", skip_header=1, delimiter=',')
        shuffle_index = np.random.permutation(all_data.shape[0])
        all_data = all_data[shuffle_index]
        each_worker_data = [nd.array(all_data[150*i:150*(i+1), 1:] * 2. - 1) for i in range(1200)]
        each_worker_label = [nd.array(all_data[150*i:150*(i+1), 0] - 1) for i in range(1200)]   
        train_data = (each_worker_data, each_worker_label)
        test_data = ((nd.array(all_data[180000:, 1:] * 2. - 1), nd.array(all_data[180000:, 0] - 1)),)
    elif args.dataset == "FEMNIST":
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
                    x = nd.array(load_dict['user_data'][user]['x']).reshape(-1, 1, 28, 28)
                    y = nd.array(load_dict['user_data'] [user]['y'])

                    each_worker_data.append(x)
                    each_worker_label.append(y)

        # random shuffle the workers
        random_order = np.random.RandomState(
            seed=args.seed).permutation(args.nworkers)
        each_worker_data = [each_worker_data[i] for i in random_order]
        each_worker_label = [each_worker_label[i] for i in random_order]
        each_worker_num = nd.array([each_worker_num[i]
                                   for i in random_order])
        train_data = (each_worker_data, each_worker_label)
        train_data_dir = os.path.join(
            "./leaf/data/femnist/data", "train")
        test_data_dir = os.path.join(
            "./leaf/data/femnist/data", "test")
        data = read_data(train_data_dir, test_data_dir)
        users, groups, train_data_ori, test_data_ori = data
        test_dataset = gluon.data.ArrayDataset(nd.concat(*[nd.array(test_data_ori[u]['x']).reshape(-1,1, 28, 28) for u in users], dim = 0), nd.concat(*[nd.array(test_data_ori[u]['y'])for u in users], dim = 0))
        test_data = gluon.data.DataLoader(test_dataset, batch_size=250, shuffle=False, last_batch='rollover')
    else: 
        raise NotImplementedError
    return train_data, test_data
    

def assign_data(train_data, bias, ctx, num_labels=10, num_workers=100, server_pc=100, p=0.1, dataset="FashionMNIST", seed=1, num_inputs=(1, 561)):
    if dataset == "purchase":
        server_data = []
        server_label = [] 
        for i in range(len(train_data[0])):
            if i >= server_pc:
                break
            rd = random.randint(1, train_data[0][i].shape[0]-1)
            server_data.append(nd.expand_dims(train_data[0][i][rd], axis = 0))
            server_label.append(train_data[1][i][rd])
        server_data = nd.concat(*server_data, dim=0) if server_pc > 0 else None
        server_label = nd.concat(*server_label, dim=0) if server_pc > 0 else None
        return server_data, server_label, train_data[0], train_data[1]
    
    elif dataset == "FEMNIST":
        server_data = []
        server_label = [] 
        for i in range(len(train_data[0])):
            if i >= server_pc:
                break
            rd = random.randint(1, train_data[0][i].shape[0]-1)
            server_data.append(nd.expand_dims(train_data[0][i][rd], axis = 0))
            server_label.append(train_data[1][i][rd])
        server_data = nd.concat(*server_data, dim=0) if server_pc > 0 else None
        server_label = nd.concat(*server_label, dim=0) if server_pc > 0 else None
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
                x = x.as_in_context(ctx).reshape(1,1,28,28)
                y = y.as_in_context(ctx)
                
                upper_bound = (y.asnumpy()) * (1. - bias) / (num_labels - 1) + bias
                lower_bound = (y.asnumpy()) * (1. - bias) / (num_labels - 1)
                rd = np.random.random_sample()
                
                if rd > upper_bound:
                    worker_group = int(np.floor((rd - upper_bound) / other_group_size) + y.asnumpy() + 1)
                elif rd < lower_bound:
                    worker_group = int(np.floor(rd / other_group_size))
                else:
                    worker_group = y.asnumpy()
                
                if server_counter[int(y.asnumpy())] < samp_dis[int(y.asnumpy())]:
                    server_data.append(x)
                    server_label.append(y)
                    server_counter[int(y.asnumpy())] += 1
                else:
                    rd = np.random.random_sample()
                    selected_worker = int(worker_group * worker_per_group + int(np.floor(rd * worker_per_group)))
                    each_worker_data[selected_worker].append(x)
                    each_worker_label[selected_worker].append(y)
                    
         
        server_data = nd.concat(*server_data, dim=0) if server_pc > 0 else None
        server_label = nd.concat(*server_label, dim=0) if server_pc > 0 else None
        
        each_worker_data = [nd.concat(*each_worker, dim=0) for each_worker in each_worker_data] 
        each_worker_label = [nd.concat(*each_worker, dim=0) for each_worker in each_worker_label]

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
                x = x.as_in_context(ctx).reshape(1, 3, 32, 32)
                y = y.as_in_context(ctx)

                upper_bound = (y.asnumpy()) * (1. - bias) / \
                    (num_labels - 1) + bias
                lower_bound = (y.asnumpy()) * (1. - bias) / (num_labels - 1)
                rd = np.random.random_sample()

                if rd > upper_bound:
                    worker_group = int(
                        np.floor((rd - upper_bound) / other_group_size) + y.asnumpy() + 1)
                elif rd < lower_bound:
                    worker_group = int(np.floor(rd / other_group_size))
                else:
                    worker_group = y.asnumpy()

                if server_counter[int(y.asnumpy())] < samp_dis[int(y.asnumpy())]:
                    server_data.append(x)
                    server_label.append(y)
                    server_counter[int(y.asnumpy())] += 1
                else:
                    rd = np.random.random_sample()
                    selected_worker = int(
                        worker_group * worker_per_group + int(np.floor(rd * worker_per_group)))
                    each_worker_data[selected_worker].append(x)
                    each_worker_label[selected_worker].append(y)

        server_data = nd.concat(*server_data, dim=0) if server_pc > 0 else None
        server_label = nd.concat(
            *server_label, dim=0) if server_pc > 0 else None

        each_worker_data = [nd.concat(*each_worker, dim=0)
                            for each_worker in each_worker_data]
        each_worker_label = [nd.concat(*each_worker, dim=0)
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
        project=f"PoisonedFL_{args.byz_type}_{args.aggregation}",
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

    ctx = get_device(args.gpu)

    # set parameters
    batch_size = args.batch_size
    num_inputs, num_outputs, num_labels = get_shapes(args.dataset)
    byz = get_byz(args.byz_type)
    num_workers = args.nworkers
    lr = args.lr
    niter = args.niter


    with ctx:
        net = get_net(args.net, num_outputs)

        net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), force_reinit=True, ctx=ctx)
        # loss
        softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

        grad_list = []
        test_acc_list = []

        # load the data
        seed = args.seed
        if seed > 0:
            mx.random.seed(seed)
            random.seed(seed)
            np.random.seed(seed)
        print("Loading data...")
        train_data, test_data = load_data(args.dataset)
        
        # assign data to the server and clients
        server_data, server_label, each_worker_data, each_worker_label = assign_data(
                                                                    train_data, args.bias, ctx, num_labels=num_labels, num_workers=num_workers, 
                                                                    server_pc=args.server_pc, p=args.p, dataset=args.dataset, seed=seed,num_inputs=num_inputs)
        print("server data shape:", server_data.shape if server_data is not None else None)
        print(each_worker_data[0].shape, each_worker_label[0].shape)
        # run a foward pass to really initialize the model
        data_count = []
        for data in each_worker_data:
            data_count.append(data.shape[0])
        net(nd.zeros(num_inputs, ctx=ctx))

        # set initial parameters
        init_model = [param.data().copy() for param in net.collect_params().values()]
        last_model = [param.data().copy() for param in net.collect_params().values()]
        history = None
        last_50_model = None
        last_grad = None
        sf = args.sf

        # set the fixed s vector for poisonedfl
        fixed_rand = nd.sign(nd.random.normal(loc=0, scale=1, shape=nd.concat(
            *[xx.reshape((-1, 1)) for xx in init_model], dim=0).shape)).squeeze()

        avg_loss = 0

        # begin training        
        for e in tqdm(range(niter)):       
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
                grad_list.append([nd.zeros_like(param.grad().copy()) for param in net.collect_params().values()])
                
            # gradients for genuine clients
            for i in participating_clients:
                ori_para = [param.data().copy() for param in net.collect_params().values()]
                for _ in range(args.local_epoch):
                    shuffled_order = np.random.choice(list(range(each_worker_data[i].shape[0])), size=each_worker_data[i].shape[0], replace=False)
                    for b_id in range(max(each_worker_data[i].shape[0]//batch_size, 1)):
                        if batch_size >= each_worker_data[i].shape[0]:
                            minibatch = list(range(each_worker_data[i].shape[0]))
                        else:
                            minibatch = shuffled_order[b_id * batch_size: (b_id +1) * batch_size]
                        with autograd.record():
                            output = net(each_worker_data[i][minibatch])
                            loss = softmax_cross_entropy(
                                output, each_worker_label[i][minibatch])
                        loss.backward()
                        avg_loss += sum(loss)/len(loss)
                        for j, (param) in enumerate(net.collect_params().values()):
                            param.set_data(param.data().copy() - lr/batch_size * param.grad().copy())
                        
                grad_list.append([( param.data().copy()- ori_data.copy()) for param, ori_data in zip(net.collect_params().values(), ori_para)])
                for param, ori_data in zip(net.collect_params().values(), ori_para):
                    param.set_data(ori_data)
            try:
                avg_loss = (avg_loss/len(participating_clients)).asnumpy()[0]
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
            elif args.aggregation == "defense1":
                return_pare_list, sf = nd_aggregation.defense1(
                    grad_list, net, lr / batch_size, parti_nfake, byz, history, fixed_rand, init_model, last_50_model, last_grad, sf, e)
            elif args.aggregation == "mean_norm":
                return_pare_list, sf = nd_aggregation.mean_norm(
                    grad_list, net, lr / batch_size, parti_nfake, byz, history,fixed_rand, init_model, last_50_model, last_grad, sf, e)
            elif args.aggregation == "specguard":
                return_pare_list, sf = nd_aggregation.specguard(
                    grad_list, net, lr / batch_size, parti_nfake, byz, history, fixed_rand, init_model, last_50_model, last_grad, sf, e,
                    server_data=server_data, server_label=server_label, loss_fn=softmax_cross_entropy, batch_size=batch_size, ctx=ctx)
            elif args.aggregation == "no":
                return_pare_list, sf = nd_aggregation.no_aggregation(
                    grad_list, net, lr / batch_size, parti_nfake, byz, history,fixed_rand, init_model, last_50_model, last_grad, sf, e)
                #raise NotImplementedError
            if parti_nfake != 0:
                if "norm" in args.aggregation:
                    last_grad = nd.mean(return_pare_list[:,:parti_nfake], axis=-1).copy()
                else:
                    last_grad = nd.mean(
                        nd.concat(*return_pare_list[:parti_nfake], dim=1), axis=-1).copy()
            del grad_list
            del return_pare_list
            grad_list = []
            current_model = [param.data().copy() for param in net.collect_params().values()]
            if (e + 1) % args.step == 0 or e + 20 >= args.niter:
                test_accuracy = evaluate_accuracy(test_data, net, ctx)
                test_acc_list.append(test_accuracy)
                print("Iteration %02d. Test_acc %0.4f" % (e, test_accuracy))
                # Log test accuracy to wandb
                wandb.log({"test/accuracy": test_accuracy, "iteration": e})
                
            if e % 50 == 0:
                last_50_model = current_model
            history = (nd.concat(*[xx.reshape((-1, 1)) for xx in current_model], dim=0) - nd.concat(*[xx.reshape((-1, 1)) for xx in last_model], dim=0) )
            last_model = [param.data().copy() for param in net.collect_params().values()]
            
            # Log scaling factor for poisonedfl attack
            if args.byz_type == "poisonedfl":
                wandb.log({"attack/scaling_factor": sf, "iteration": e})

            from os import path
        
        # Log final test accuracy
        final_test_acc = evaluate_accuracy(test_data, net, ctx)
        wandb.log({"test/final_accuracy": final_test_acc})
        print("Final Test Accuracy: %0.4f" % final_test_acc)
        
        # Finish wandb run
        wandb.finish()
                
        del test_acc_list
        test_acc_list = []

   
if __name__ == "__main__":
    args = parse_args()
    main(args)
