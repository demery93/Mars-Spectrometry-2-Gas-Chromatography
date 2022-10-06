import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
from utils import print_stats, timeit_context
from dataset import MarsSpectrometryDataset, render_image
from config import config


def check_render_image():
    sample_id = 'S0001'


    t = pd.read_feather(f'processed/features/{sample_id}.f')
    step_pos = np.where(np.diff(t['mass'].values, prepend=0) < 0)[0]

    intensity = t['intensity'].values
    intensity = intensity / intensity.max()

    mass = t['mass'].values.astype(int)
    time = t.time.values
    #time = -1 * (time - np.max(time)) + 0.01
    #print(time)

    with timeit_context('render img'):
        for i in range(100):
            p = render_image(time=time, mass=mass, intensity=intensity, step_pos=step_pos, time_query_range=10)

    p[4, :] *= 0.1
    p = p / p.max()

    p = np.clip(p, 1e-5, 1.0)
    p = np.log10(p)
    plt.imshow(p)
    plt.xlabel('Mass')
    plt.ylabel('Time')
    plt.show()

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
        plt.xlabel('Mass')
        plt.ylabel('Time')
        plt.show()


def check_aug():
    ds = MarsSpectrometryDataset(
        fold=0, is_training=True,
        dataset_type='train',
        max_mass=250,
        min_mass=13,
        max_time=50)

    sample_id = 'S0491'

    for i in range(100):
        sample = ds[ds.sample_ids.index(sample_id)]
        print(sample['item'], sample['sample_id'], sample['label'])
        print_stats('image', sample['image'])

        plt.imshow(sample['image'])
        plt.show()


def check_performance():
    ds = MarsSpectrometryDataset(
        fold=0, is_training=True,
        dataset_type='train',
        max_mass=250,
        min_mass=13,
        max_time=50)

    print()
    with timeit_context('predict 100'):
        for i, sample in tqdm(enumerate(ds), total=len(ds)):
            pass


if __name__ == '__main__':
    check_render_image()
    #check_dataset()
    #check_aug()
    #check_performance()