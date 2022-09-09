import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
from utils import print_stats

import config

def render_image(time: np.ndarray,
                 mass: np.ndarray,
                 intensity: np.ndarray,
                 step_pos: np.ndarray,
                 time_step=1,
                 time_query_range=6,
                 prob_smooth=6,
                 max_mass=250,
                 min_mass=13
                 ):

    # temp_step = 6
    max_time_id = (config.max_time - config.min_time) // time_step
    nions = (config.max_mass - config.min_mass)

    # temp_query_range = 10
    prob_smooth=6

    res = np.zeros((nions, config.max_time), dtype=np.float32) - 1
    step_time = [np.mean(v) for v in np.split(time, step_pos)][1:-1]
    time_bands = [[] for t in range(config.max_time + time_query_range + 1)]  # temp: list of steps
    for step, t in enumerate(step_time):
        t = math.floor(t)
        if 0 <= t < len(time_bands):
            time_bands[t].append(step)

    for time_id in range(max_time_id):
        t = config.min_time + time_id * time_step
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
            res[mass[i] - config.min_mass, time_id] = intensity[i]

    return res.T


def check_render_image():
    sample_id = 'S0754'
    sample_id = 'S0001'


    t = pd.read_csv(f'../data/train_features_pp/{sample_id}.csv')
    step_pos = np.where(np.diff(t['m/z'].values, prepend=0) < 0)[0]

    abd = t['intensity_sub_q20'].values
    abd = abd / abd.max()

    mz = t['m/z'].values.astype(int)

    with utils.timeit_context('render img'):
        for i in range(100):
            p = render_image(temp=t['temp'].values, mz=mz, abd=abd, step_pos=step_pos)

    p[4, :] *= 0.1
    p = p / p.max()

    p = np.clip(p, 1e-5, 1.0)
    p = np.log10(p)

    # plt.hist(p)
    # plt.figure()

    plt.imshow(p)
    plt.show()


class MarsSpectrometryDataset(tf.keras.utils.Sequence):
    def __init__(self, fold, is_training,
                 dataset_type='train',
                 max_mass=250,
                 min_mass=13,
                 max_time=50
                 ):

        self.max_time = max_time
        self.max_mass = max_mass
        self.min_mass = min_mass
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
        prob_sub = np.random.rand(4)
        prob_sub = prob_sub / prob_sub.sum()

        sub_bg = False
        if self.sub_bg_prob > 0:
            sub_bg = np.random.rand() < self.sub_bg_prob

        if sub_bg:
            intensity = (
                    t['intensity_sub_bg_sub_min'].values * prob_sub[0] +
                    t['intensity_sub_bg_sub_q5'].values * prob_sub[1] +
                    t['intensity_sub_bg_sub_q10'].values * prob_sub[2] +
                    t['intensity_sub_bg_sub_q20'].values * prob_sub[3]
            )
        else:
            intensity = (
                    t['intensity_sub_min'].values * prob_sub[0] +
                    t['intensity_sub_q5'].values * prob_sub[1] +
                    t['intensity_sub_q10'].values * prob_sub[2] +
                    t['intensity_sub_q20'].values * prob_sub[3]
            )

        mass = t['mass'].values.astype(int)
        p = render_image(time=t['time'].values, mass=mass, intensity=intensity, step_pos=step_pos,
                         time_step=self.time_step,
                         time_query_range=self.time_query_range,
                         prob_smooth=self.prob_smooth,
                         max_time=self.max_time,
                         max_mass=self.max_mass
                         )

        return p

    def __getitem__(self, item):
        sample_id = self.sample_ids[item]
        labels = self.labels.loc[sample_id]
        label_values = [labels[k] for k in config.targetcols]
        label_values = np.array(label_values, dtype=np.float32)
        metadata = self.metadata.loc[sample_id]

        p = self.render_item(sample_id)

        p = p.astype(np.float32)

        res = {
            'item': item,
            'image': p,
            'sample_id': sample_id,
            'label': label_values,
            'derivatized': metadata['derivatized']
        }

        return res


def check_dataset():
    ds = MarsSpectrometryDataset(
        fold=0, is_training=True,
        dataset_type='train',
        max_mass=250,
        min_mass=13,
        max_time=50)

    ds.sample_ids = ds.sample_ids[::-1]

    for i, sample in enumerate(ds):
        print(sample['item'], sample['sample_id'], sample['derivatized'], sample['label'])
        print_stats('image', sample['image'])

        plt.imshow(sample['image'])
        plt.xlabel('T')
        plt.ylabel('m/z')
        plt.show()


def check_aug():
    ds = MarsSpectrometryDataset(
        fold=0, is_training=True, output_pytorch_tensor=False,
        he_scale=0.1,
        he_var=0.0,
        temp_step=12,
        temp_query_range=16,
        prob_smooth=12,
        norm_to_one=False,
        min_clip=1e-4,
        mix_aug_prob=0.0,
        mz_var=1.0,
        sub_bg_prob=0.5
    )

    sample_id = 'S0491'

    for i in range(100):
        sample = ds[ds.sample_ids.index(sample_id)]
        print(sample['item'], sample['sample_id'], sample['sample_id2'], sample['label'], sample['label2'])
        utils.print_stats('image', sample['image'])

        plt.imshow(sample['image'])
        plt.show()


def check_performance():
    ds = MarsSpectrometryDataset(fold=0, is_training=True, output_pytorch_tensor=False,
                                 he_scale=0.1,
                                 he_var=0.0,
                                 temp_step=12,
                                 temp_query_range=16,
                                 prob_smooth=12,
                                 norm_to_one=False,
                                 min_clip=1e-4,
                                 mix_aug_prob=0.0,
                                 mz_var=1.0,
                                 sub_bg_prob=0.5
                                 )
    print()
    with utils.timeit_context('predict 100'):
        for i, sample in tqdm(enumerate(ds), total=len(ds)):
            pass

#train_gen = MarsSpectrometryDataset(0, True, dataset_type='train',max_mass=250,min_mass=13,max_time=50)
#val_imgs = train_gen.load_val()
if __name__ == '__main__':
    # check_render_image()
    check_dataset()
    # check_aug()
    # check_performance()