import os
# Ours
os.system("python test_agr_pytorch.py --dataset cifar10 --gpu 3 --net cnn_cifar --niter 10000 --nworkers 1200 --nfake 240 --aggregation specguard2 --byz_type init_attack --sf 8 --local_epoch 1 --bias 0.2  --lr 0.03")

# os.system("python test_agr_pytorch.py --dataset cifar10 --gpu 4 --net cnn_cifar --niter 10000 --nworkers 1200 --nfake 240 --aggregation trim --byz_type poisonedfl --sf 8 --local_epoch 1 --bias 0.2  --lr 0.03")

# os.system("python test_agr_pytorch.py --dataset cifar10 --gpu 5 --net cnn_cifar --niter 10000 --nworkers 1200 --nfake 240 --aggregation mean_norm --byz_type poisonedfl --sf 8 --local_epoch 1 --bias 0.2  --lr 0.03")
