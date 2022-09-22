import pandas as pd
import numpy as np
import gc
import einops
from sklearn.model_selection import KFold
from config import config
from model_definitions import cnn, lstm, cnn2d
import tensorflow as tf
from utils import aggregated_log_loss
from einops.layers.keras import Rearrange
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import math
from dataset import MarsSpectrometryDataset
from utils import aggregated_log_loss

labels = pd.read_csv("input/train_labels.csv")
sub = pd.read_csv("input/submission_format.csv")
labels.set_index('sample_id', inplace=True)
oof = labels.copy()

experiment = {"time_step":1,
              "max_time": 50,
              "timesteps":50,
              "max_mass":250,
              "min_mass":1,
              "nions":249}

for i in range(config.n_folds):
    model = lstm(experiment['timesteps'], experiment['nions'])
    train_ds = MarsSpectrometryDataset(
        fold=i,
        is_training=True,
        dataset_type='train',
        time_step=experiment['time_step'],
        max_time=experiment['max_time'],
        max_mass=experiment['max_mass'],
        min_mass=experiment['min_mass'])

    val_ds = MarsSpectrometryDataset(
        fold=i,
        is_training=False,
        dataset_type='train',
        time_step=experiment['time_step'],
        max_time=experiment['max_time'],
        max_mass=experiment['max_mass'],
        min_mass=experiment['min_mass'])

    callback = [tf.keras.callbacks.EarlyStopping(patience=5),
                tf.keras.callbacks.ModelCheckpoint(f'cnn_{i}.h5', save_best_only=True, save_weights_only=True),
                tf.keras.callbacks.ReduceLROnPlateau(patience=3)]
    history = model.fit(
        train_ds,
        verbose=1,
        epochs=100,
        batch_size=8,
        validation_data=val_ds,
        callbacks=callback
    )
    model.load_weights(f'cnn_{i}.h5')
    val_pred = np.zeros((len(val_ds.sample_ids), 9))
    for i in range(config.tta):
        val_pred += model.predict(val_ds) / config.tta

    oof.loc[val_ds.sample_ids] = val_pred

    del train_ds, val_ds, model, history
    gc.collect()


y = labels.values
oof = oof.values
print(aggregated_log_loss(y, oof)) #0.18321898071318965

test_ds = MarsSpectrometryDataset(
    fold=0,
    is_training=False,
    dataset_type='test_val',
    time_step=experiment['time_step'],
    max_time=experiment['max_time'],
    max_mass=experiment['max_mass'],
    min_mass=experiment['min_mass'])

preds = np.zeros(sub[config.targetcols].shape)
for i in range(config.n_folds):
    model = lstm(experiment['timesteps'], experiment['nions'])
    model.load_weights(f'cnn_{i}.h5')
    pred = np.zeros(preds.shape)
    for j in range(config.tta):
        pred += model.predict(test_ds) / config.tta

    preds += pred / config.n_folds

sub[config.targetcols] = preds
sub.to_csv("output/submission.csv", index=False, header=True)