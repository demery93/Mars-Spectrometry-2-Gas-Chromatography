import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
from utils import print_stats, timeit_context, load_config_data
from dataset import MarsSpectrometryDataset, render_image
from config import config


if __name__ == '__main__':
    experiment_name = "130_resnet34"
    cfg = load_config_data(experiment_name)

    model_params = cfg['model_params']
    dataset_params = cfg['dataset_params']
    train_params = cfg['train_params']
    predict_params = cfg['predict_params']
    ds = MarsSpectrometryDataset(
        fold=0,
        is_training=True,
        dataset_type='train',
        batch_size=1,
        **dataset_params)
    p = ds.render_item('S0003')
    plt.imshow(p)
    plt.xlabel('Mass')
    plt.ylabel('Time')
    plt.show()