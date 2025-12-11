import os
# Ours
os.system("python test_agr_pytorch.py --dataset purchase --gpu 3 --net dnn --niter 6000 --nworkers 1200 --nfake 240 --aggregation specguard --byz_type lie_attack --sf 8 --local_epoch 1 --batch_size 128 --lr 0.1  ")

# os.system("python test_agr.py --dataset purchase --gpu 1 --net dnn --niter 6000 --nworkers 1200 --nfake 240 --aggregation trim --byz_type poisonedfl --sf 8 --local_epoch 1 --batch_size 128 --lr 0.1  ")

# os.system("python test_agr.py --dataset purchase --gpu 2 --net dnn --niter 6000 --nworkers 1200 --nfake 240 --aggregation mean_norm --byz_type poisonedfl --sf 8 --local_epoch 1 --batch_size 128 --lr 0.1  ")
