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

    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)

    out = tf.keras.layers.Dense(len(targetcols), activation='sigmoid')(x)
    model = tf.keras.models.Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

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


kfold = KFold(n_splits=config.n_fold, random_state=config.seed, shuffle=True)
oof = np.zeros((len(labels), len(targetcols)))
val_pred = []
test_pred = []
# Iterate through each fold
scores = []
from tqdm import tqdm

x_l, y_l = [], []
for sample in tqdm(labels.sample_id.values):
    df = pd.read_feather(f"processed/features/{sample}.f")
    y_tmp = labels[labels.sample_id == sample]
    x, y = create_dataset_part(df, y_tmp[targetcols].values, scale=1)
    x_l.append(x)
    y_l.append(y)

Xtrain = np.concatenate(x_l, axis=0)
Ytrain = np.concatenate(y_l)
del x_l, y_l
gc.collect()

for fold, (trn_ind, val_ind) in enumerate(kfold.split(labels, labels)):
    train_samples, val_samples = labels.sample_id.values[trn_ind], labels.sample_id.values[val_ind]

    #train_set = train_generator(train_samples, labels)
    #val_set = train_generator(val_samples, labels)
    x_train, y_train = Xtrain[trn_ind], Ytrain[trn_ind]
    x_val, y_val = Xtrain[val_ind], Ytrain[val_ind]
    model = cnn()
    callback = [tf.keras.callbacks.EarlyStopping(patience=5),
                tf.keras.callbacks.ModelCheckpoint(f'cnn_{fold}.h5', save_best_only=True, save_weights_only=True),
                tf.keras.callbacks.ReduceLROnPlateau(patience=3)]
    #model.fit(train_set, steps_per_epoch=len(train_samples), epochs=100, verbose=1, validation_data=val_set, validation_steps=len(val_samples), callbacks=callback)
    model.fit(x_train, y_train, batch_size=16, epochs=100, verbose=1, validation_data=(x_val, y_val), callbacks=callback)
    oof[val_ind] = model.predict(x_val)

y = labels[targetcols].values
score = np.round(aggregated_log_loss(y, oof), 3)
print(f"CV Score: {aggregated_log_loss(y, oof)}")
# CV Score: 0.19835358760840024

val_pred = np.mean(np.dstack(val_pred), axis=-1)
test_pred = np.mean(np.dstack(test_pred), axis=-1)
print(val_pred.shape, test_pred.shape)
sub = pd.read_csv("input/submission_format.csv")

pred = np.concatenate([val_pred, test_pred], axis=0)
pred = pd.DataFrame(pred, columns=targetcols)
for c in targetcols:
    sub[c] = pred[c]

sub.to_csv(f"output/submission_{score}.csv", index=False, header=True)
for i in range(len(targetcols)):
    print(oof[:,i].mean(), sub[targetcols].values[:,i].mean())