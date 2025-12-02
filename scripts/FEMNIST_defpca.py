import os
# Ours
#os.system("python test_agr.py --dataset FEMNIST --gpu -1 --net cnn --niter 10000 --nworkers 1200 --nfake 240 --aggregation specguard --byz_type poisonedfl --sf 8 --local_epoch 1")

os.system("python test_agr.py --dataset FEMNIST --gpu -1 --net cnn --niter 10000 --nworkers 1200 --nfake 240 --server_pc 144 --aggregation specguard --byz_type poisonedfl --sf 8 --local_epoch 1")


#os.system("python test_agr.py --dataset FEMNIST --gpu 0 --net cnn --niter 10000 --nworkers 1200 --nfake 240 --aggregation trim --byz_type poisonedfl --sf 8 --local_epoch 1")

#os.system("python test_agr.py --dataset FEMNIST --gpu 0 --net cnn --niter 10000 --nworkers 1200 --nfake 240 --aggregation mean_norm --byz_type poisonedfl --sf 8 --local_epoch 1")
