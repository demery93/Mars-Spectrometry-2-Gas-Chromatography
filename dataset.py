import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
from utils import print_stats, timeit_context

from config import config

def render_image(time: np.ndarray,
                 mass: np.ndarray,
                 intensity: np.ndarray,
                 step_pos: np.ndarray,
                 time_step=1
                 ):
    max_time_id = int((config.max_time - config.min_time) // time_step)
    nions = (config.max_mass - config.min_mass)

    res = np.zeros((nions, config.max_time), dtype=np.float32) - 1
    step_time = [np.mean(v) for v in np.split(time, step_pos)][1:-1]
    time_bands = [[] for t in range(int(config.max_time // time_step) + 1)]  # temp: list of steps
    for step, t in enumerate(step_time):
        t = int(t // time_step)
        if 0 <= t < len(time_bands):
            time_bands[t].append(step)

    #sum = np.max(intensity)
    for time_id in range(max_time_id):
        t = int(config.min_time + time_id * time_step)
        src_steps = []
        for band in time_bands[max(0, t):t + 1]:
            for step in band:
                src_steps.append(step)

        if not len(src_steps):
            continue

        src_step = np.random.choice(src_steps)

        sum = np.sum(intensity[step_pos[src_step]:step_pos[src_step+1]])
        for i in range(step_pos[src_step], step_pos[src_step + 1]):
            res[mass[i] - config.min_mass, time_id] = intensity[i]

    res = res.T
    res = np.clip(res, 0, 1e9)
    sum = res.sum(axis=0).reshape((1,-1))
    sum[sum == 0] = 1

    res = res / sum
    return res

class MarsSpectrometryDataset(tf.keras.utils.Sequence):
    def __init__(self, fold, is_training,
                 dataset_type='train',
                 time_step=1,
                 time_query_range=0,
                 max_mass=250,
                 min_mass=13,
                 max_time=50
                 ):

        self.max_time = max_time
        self.max_mass = max_mass
        self.min_mass = min_mass
        self.time_step = time_step
        self.time_query_range = time_query_range
        self.sample_ids = []
        self.metadata = pd.read_csv("input/metadata.csv", index_col='sample_id')

        if dataset_type == 'train':
            folds = pd.read_csv('processed/folds.csv')

            fold = int(fold)
            if is_training:
                folds = folds[folds.fold != fold]
            else:
                folds = folds[folds.fold == fold]

            self.sample_ids = list(folds.sample_id.values)
            self.labels = pd.read_csv('input/train_labels.csv', index_col='sample_id')
        elif dataset_type == 'val':
            self.sample_ids = list(self.metadata[self.metadata.split == 'val'].index.values)
        elif dataset_type == 'test_val':
            self.sample_ids = list(self.metadata[self.metadata.split != 'train'].index.values)
            self.labels = pd.DataFrame(index=self.sample_ids)
            for col in config.targetcols:
                self.labels[col] = 0.0

        self.samples_data = {
            sample_id: pd.read_feather(f'processed/features/{sample_id}.f')
            for sample_id in self.sample_ids
        }

        print(f'Training {is_training}, samples: {len(self.sample_ids)} ')

    def __len__(self):
        return len(self.sample_ids)

    def render_item(self, sample_id):
        t = self.samples_data[sample_id]
        step_pos = np.where(np.diff(t['mass'].values, prepend=0) < 0)[0]
        intensity = t['intensity_sub_min'].values
        mass = t['mass'].values.astype(int)

        p = render_image(time=t['time'].values,
                         mass=mass,
                         intensity=intensity,
                         step_pos=step_pos,
                         time_step=self.time_step
                         )
        p[:, :] *= 2 ** np.random.normal(loc=0, scale=0.5, size=(p.shape[0], 1))
        return p

    def __getitem__(self, item):
        sample_id = self.sample_ids[item]
        labels = self.labels.loc[sample_id]
        label_values = [labels[k] for k in config.targetcols]
        label_values = np.array(label_values, dtype=np.float32)
        metadata = self.metadata.loc[sample_id]

        p = self.render_item(sample_id)
        p = p.astype(np.float32)

        return (p.reshape((1, 50, 237)), label_values.reshape((1,9)))