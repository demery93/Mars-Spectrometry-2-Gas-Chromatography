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
from dataset import MarsSpectrometryDataset
from utils import aggregated_log_loss


def cnn():
    inp = tf.keras.layers.Input(shape=(50, 237))

    c1 = tf.keras.layers.Conv1D(50, 2, strides=1, dilation_rate=2 ** 0, padding='same')(inp)
    c1 = tf.keras.layers.BatchNormalization()(c1)
    c1 = tf.keras.layers.Activation('relu')(c1)
    c2 = tf.keras.layers.Conv1D(50, 2, strides=1, dilation_rate=2 ** 1, padding='same')(inp)
    c2 = tf.keras.layers.BatchNormalization()(c2)
    c2 = tf.keras.layers.Activation('relu')(c2)
    c3 = tf.keras.layers.Conv1D(50, 2, strides=1, dilation_rate=2 ** 2, padding='same')(inp)
    c3 = tf.keras.layers.BatchNormalization()(c3)
    c3 = tf.keras.layers.Activation('relu')(c3)
    c4 = tf.keras.layers.Conv1D(50, 2, strides=1, dilation_rate=2 ** 3, padding='same')(inp)
    c4 = tf.keras.layers.BatchNormalization()(c4)
    c4 = tf.keras.layers.Activation('relu')(c4)
    c5 = tf.keras.layers.Conv1D(50, 2, strides=1, dilation_rate=2 ** 4, padding='same')(inp)
    c5 = tf.keras.layers.BatchNormalization()(c5)
    c5 = tf.keras.layers.Activation('relu')(c5)
    c = tf.keras.layers.concatenate([c1, c2, c3, c4, c5])

    x = tf.keras.layers.Flatten()(c)
    x = tf.keras.layers.Dropout(0.5)(x)
    #x = tf.keras.layers.BatchNormalization()(x)
    #x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)

    out = tf.keras.layers.Dense(len(config.targetcols), activation='sigmoid')(x)
    model = tf.keras.models.Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

def cnn():
    inp = tf.keras.layers.Input(shape=(50, 237))

    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100, return_sequences=False))(inp)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)

    out = tf.keras.layers.Dense(len(config.targetcols), activation='sigmoid')(x)
    model = tf.keras.models.Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model


labels = pd.read_csv("input/train_labels.csv")
sub = pd.read_csv("input/submission_format.csv")
labels.set_index('sample_id', inplace=True)
oof = labels.copy()

for i in range(config.n_folds):
    model = cnn()
    train_ds = MarsSpectrometryDataset(
        fold=i, is_training=True, time_step=1,
        dataset_type='train',
        max_time=50)

    val_ds = MarsSpectrometryDataset(
        fold=i, is_training=False, time_step=1,
        dataset_type='train',
        max_time=50)

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
    oof.loc[val_ds.sample_ids] = model.predict(val_ds)

    del train_ds, val_ds, model, history
    gc.collect()


y = labels.values
oof = oof.values
print(aggregated_log_loss(y, oof)) #0.1982197260925508

test_ds = MarsSpectrometryDataset(
    fold=0, is_training=False, time_step=1,
    dataset_type='test_val',
    max_time=50)

preds = np.zeros(sub[config.targetcols].shape)
for i in range(config.n_folds):
    model = cnn()
    model.load_weights(f'cnn_{i}.h5')
    preds += model.predict(test_ds) / config.n_folds

sub[config.targetcols] = preds
sub.to_csv("output/submission.csv", index=False, header=True)