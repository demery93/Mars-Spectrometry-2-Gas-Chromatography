import pandas as pd
import numpy as np
import gc
import einops
from sklearn.model_selection import KFold
from config import config
import tensorflow as tf
from utils import aggregated_log_loss
from einops.layers.keras import Rearrange
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import math

TIMESTEPS = 49
labels = pd.read_csv("input/train_labels.csv")
metadata = pd.read_csv("input/metadata.csv")

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
    time = df.time.values
    mass = np.round(df.mass).values.astype(int)
    intensity = df.intensity.values
    step_pos = step_pos
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


targetcols = ['aromatic', 'hydrocarbon', 'carboxylic_acid',
       'nitrogen_bearing_compound', 'chlorine_bearing_compound',
       'sulfur_bearing_compound', 'alcohol', 'other_oxygen_bearing_compound',
       'mineral']



def train_generator(sample_ids, targets):
    while 1:
        for sample_id in np.random.permutation(sample_ids):
            df = pd.read_csv(f"processed/features/{sample_id}.csv")
            y_tmp = targets[targets.sample_id == sample_id]
            scale = np.random.normal(loc=0.0, scale=1.0, size=None)
            percentile = np.random.randint(4)
            yield create_dataset_part(df, y_tmp[targetcols].values,  scale=scale)
            gc.collect()

def create_dataset_part(df, y, scale=1):
    step_pos = np.where(np.diff(df['mass'].values, prepend=0) < 0)[0]
    x = render_image(df.time.values,
                     np.round(df.mass).values.astype(int),
                     df.intensity.values,
                     step_pos)


    x = np.nan_to_num(x)
    x = scale*x

    x = x.reshape((-1, 50, 237))

    sum_vals = np.sum(np.clip(x, 0, 1e12), axis=2)
    sum_vals[sum_vals == 0] = 1
    x = x / sum_vals.reshape((-1, 50, 1))

    return(x, y)



df = pd.read_feather(f"processed/features/S0000.f")
y_tmp = labels[labels.sample_id == 'S0000']
x, y = create_dataset_part(df, y_tmp[targetcols].values, scale=1)
