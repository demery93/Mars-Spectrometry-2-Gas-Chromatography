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
                 time_step=1,
                 min_time = 0,
                 max_time = 50,
                 min_mass = 1,
                 max_mass = 250,
                 time_query_range = 0,
                 prob_smooth = 6
                 ):
    max_time_id = int((max_time - min_time) // time_step)
    nions = (max_mass - min_mass)

    res = np.zeros((int((max_time - min_time)//time_step), nions), dtype=np.float32) - 1
    step_time = [np.mean(v) for v in np.split(time, step_pos)][1:-1]
    time_bands = [[] for t in range(int(max_time // time_step) + time_query_range + 1)]  # temp: list of steps
    for step, t in enumerate(step_time):
        t = int(t // time_step)
        if 0 <= t < len(time_bands):
            time_bands[t].append(step)

    for time_id in range(max_time_id):
        t = int(config.min_time + time_id * time_step)
        src_steps = []
        src_steps_p = []
        for band in time_bands[max(0, t - time_query_range):t + time_query_range + 1]:
            for step in band:
                src_steps.append(step)
                src_steps_p.append(1.0 / (prob_smooth + abs(t - step_time[step])))

        if not len(src_steps):
            continue

        src_steps_p = np.array(src_steps_p)
        src_step = np.random.choice(src_steps, p=src_steps_p / src_steps_p.sum())

        for i in range(step_pos[src_step], step_pos[src_step + 1]):
            res[time_id, mass[i] - config.min_mass] = intensity[i]

    return res

def render_sequence(time: np.ndarray,
                 mass: np.ndarray,
                 intensity: np.ndarray,
                 time_pos: np.ndarray,
                 min_mass = 1,
                 max_mass = 250,
                 sequence_length = 2000
                 ):

    nions = int(max_mass - min_mass)
    res = np.zeros((sequence_length, nions), dtype=np.float32) - 1
    t = np.zeros((sequence_length,1), dtype=np.float32) - 1
    time_length = min(len(time_pos), sequence_length)
    for time_id in range(time_length-1):
        res[time_id, mass[time_pos[time_id]:time_pos[time_id+1]]-min_mass] = intensity[time_pos[time_id]:time_pos[time_id+1]]

    t[:time_length, 0] = time[time_pos[:time_length]]
    return res, t


class MarsSpectrometryDataset(tf.keras.utils.Sequence):
    def __init__(self, fold, is_training,
                 dataset_type='train',
                 time_step=1,
                 time_query_range=0,
                 max_mass=250,
                 min_mass=13,
                 max_time=50,
                 norm_to_one=False,
                 batch_size=1,
                 log_space=True,
                 min_clip=0.01,
                 norm_max=True,
                 norm_sum=False,
                 prob_smooth=6,
                 make_3d=False
                 ):

        self.max_time = max_time
        self.max_mass = max_mass
        self.min_mass = min_mass
        self.time_step = time_step
        self.time_query_range = time_query_range
        self.nions = max_mass - min_mass
        self.timesteps = int(max_time // time_step)
        self.norm_to_one = norm_to_one
        self.log_space = log_space
        self.min_clip = min_clip
        self.norm_max = norm_max
        self.norm_sum = norm_sum
        self.prob_smooth = prob_smooth
        self.make_3d = make_3d
        self.sample_ids = []
        self.metadata = pd.read_csv("input/metadata.csv", index_col='sample_id')
        self.batch_size = batch_size

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
            self.labels = pd.read_csv('input/val_labels.csv', index_col='sample_id')
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
        return len(self.sample_ids) // self.batch_size

    def render_item(self, sample_id):
        t = self.samples_data[sample_id]
        step_pos = np.where(np.diff(t['mass'].values, prepend=0) < 0)[0]
        mass = t['mass'].values.astype(int)

        prob_sub = np.random.rand(4)
        prob_sub = prob_sub / prob_sub.sum()

        intensity = (t['intensity_sub_min'].values * prob_sub[0] +
                     t['intensity_sub_q5'].values * prob_sub[1] +
                     t['intensity_sub_q10'].values * prob_sub[2] +
                     t['intensity_sub_q20'].values * prob_sub[3]
                     )


        p = render_image(time=t['time'].values,
                         mass=mass,
                         intensity=intensity,
                         step_pos=step_pos,
                         time_step=self.time_step,
                         max_time = self.max_time,
                         min_mass = self.min_mass,
                         max_mass = self.max_mass,
                         time_query_range = self.time_query_range,
                         prob_smooth = self.prob_smooth
                         )

        if self.norm_max:
            p = p / p.max()
        p = p * (2 ** np.random.normal(0, 0.05))
        p = np.clip(p, self.min_clip, 1e10)

        if self.log_space:
            p = np.log10(p)
            if self.norm_to_one:
                p = 1.0 + p / abs(np.log10(self.min_clip))

        if(self.norm_sum):
            s = p.sum(axis=0).reshape((1, -1))
            s[s == 0] = 1
            p = p / s

        return p

    def __getitem__(self, item):
        batch = self.sample_ids[item * self.batch_size:(item + 1) * self.batch_size]
        labels = self.labels.loc[batch]
        label_values = [labels[k] for k in config.targetcols]
        label_values = np.array(label_values, dtype=np.float32)
        metadata = self.metadata.loc[batch]
        X = self.__get_data(batch)
        if(self.make_3d):
            t = (np.arange(self.timesteps) / self.timesteps)[None, :, None, None]
            m = (np.arange(self.nions) / self.nions)[None, None, :, None]
            t = t.repeat(X.shape[0], axis=0)
            t = t.repeat(self.nions, axis=2)
            m = m.repeat(X.shape[0], axis=0)
            m = m.repeat(self.timesteps, axis=1)
            X = np.concatenate([X[:,:,:,None], t, m], axis=3)
        return X, label_values.T

    def __get_data(self, batch):
        X=[]
        for sample_id in batch:
            p = self.render_item(sample_id)
            p = p.astype(np.float32)
            X.append(p)

        X = np.asarray(X)
        return X

    def load_val(self):
        X=[]
        for sample_id in self.sample_ids:
            p = self.render_item(sample_id)
            p = p.astype(np.float32)
            X.append(p)

        X = np.asarray(X)
        if(self.make_3d):
            t = (np.arange(self.timesteps) / self.timesteps)[None, :, None, None]
            m = (np.arange(self.nions) / self.nions)[None, None, :, None]
            t = t.repeat(X.shape[0], axis=0)
            t = t.repeat(self.nions, axis=2)
            m = m.repeat(X.shape[0], axis=0)
            m = m.repeat(self.timesteps, axis=1)
            X = np.concatenate([X[:,:,:,None], t, m], axis=3)
        return X


class MarsSpectrometryDatasetSequence(tf.keras.utils.Sequence):
    def __init__(self, fold, is_training,
                 dataset_type='train',
                 max_mass=250,
                 min_mass=13,
                 norm_to_one=False,
                 batch_size=1,
                 log_space=True,
                 min_clip=0.01,
                 norm_max=True,
                 norm_sum=False,
                 mz_dropout=0.00,
                 time_dropout=0.00
                 ):

        self.max_mass = max_mass
        self.min_mass = min_mass
        self.nions = max_mass - min_mass
        self.norm_to_one = norm_to_one
        self.log_space = log_space
        self.min_clip = min_clip
        self.norm_max = norm_max
        self.norm_sum = norm_sum
        self.mz_dropout = mz_dropout
        self.time_dropout = time_dropout
        self.sample_ids = []
        self.metadata = pd.read_csv("input/metadata.csv", index_col='sample_id')
        self.batch_size = batch_size

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
            self.labels = pd.read_csv('input/val_labels.csv', index_col='sample_id')
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
        return len(self.sample_ids) // self.batch_size

    def render_item(self, sample_id):
        t = self.samples_data[sample_id]
        time_pos = np.where(np.diff(t['time'].values, prepend=0) != 0)[0]
        mass = t['mass'].values.astype(int)

        prob_sub = np.random.rand(4)
        prob_sub = prob_sub / prob_sub.sum()

        intensity = (t['intensity_sub_min'].values * prob_sub[0] +
                     t['intensity_sub_q5'].values * prob_sub[1] +
                     t['intensity_sub_q10'].values * prob_sub[2] +
                     t['intensity_sub_q20'].values * prob_sub[3]
                     )


        p, t = render_sequence(time=t['time'].values,
                         mass=mass,
                         intensity=intensity,
                         time_pos=time_pos,
                         min_mass=self.min_mass,
                         max_mass=self.max_mass
                         )

        if self.norm_max:
            p = p / p.max()
        p = p * (2 ** np.random.normal(0, 0.05))
        p = np.clip(p, self.min_clip, 1e10)

        if self.log_space:
            p = np.log10(p)
            if self.norm_to_one:
                p = 1.0 + p / abs(np.log10(self.min_clip))

        if(self.norm_sum):
            s = p.sum(axis=0).reshape((1, -1))
            s[s == 0] = 1
            p = p / s

        mz_inds = np.where(np.random.rand(p.shape[1]) < self.mz_dropout)
        time_inds = np.where(np.random.rand(p.shape[0]) < self.mz_dropout)
        p[:,mz_inds] = 0
        p[time_inds, :] = 0

        t = t / 50


        return p, t

    def __getitem__(self, item):
        batch = self.sample_ids[item * self.batch_size:(item + 1) * self.batch_size]
        labels = self.labels.loc[batch]
        label_values = [labels[k] for k in config.targetcols]
        label_values = np.array(label_values, dtype=np.float32)
        metadata = self.metadata.loc[batch]
        X = self.__get_data(batch)
        return X, label_values.T

    def __get_data(self, batch):
        X1, X2 = [], []
        for sample_id in batch:
            p, t = self.render_item(sample_id)
            p = p.astype(np.float32)
            X1.append(p)
            X2.append(t)

        X1 = np.asarray(X1)
        X2 = np.asarray(X2)
        return [X1, X2]

    def load_val(self):
        X1, X2 = [], []
        for sample_id in self.sample_ids:
            p, t = self.render_item(sample_id)
            p = p.astype(np.float32)
            X1.append(p)
            X2.append(t)

        X1 = np.asarray(X1)
        X2 = np.asarray(X2)
        return [X1, X2]