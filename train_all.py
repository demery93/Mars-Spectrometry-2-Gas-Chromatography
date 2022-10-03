import os
import sys

experiments = ['100_conv1d']

for exp in experiments:
    cmd = f"python train_model.py 0 {exp}"
    os.system(cmd) #0.1828

    cmd = f"python train_model.py 1 {exp}"
    os.system(cmd) #0.1744

    cmd = f"python train_model.py 2 {exp}"
    os.system(cmd) #0.1772

    cmd = f"python train_model.py 3 {exp}"
    os.system(cmd) #0.1863