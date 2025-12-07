import os

if not os.path.exists("purchase"):
    os.mkdir("purchase")
os.system("wget https://www.comp.nus.edu.sg/~reza/files/dataset_purchase.tgz -O purchase/dataset_purchase.tgz")
os.system("tar -xzf purchase/dataset_purchase.tgz -C purchase")