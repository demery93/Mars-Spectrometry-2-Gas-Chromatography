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

TIMESTEPS = 49
train = pd.read_feather("processed/train.f")
val = pd.read_feather("processed/val.f")
test = pd.read_feather("processed/test.f")
labels = pd.read_csv("input/train_labels.csv")
metadata = pd.read_feather("processed/metadata.f")
sub = pd.read_csv("input/submission_format.csv")
labels = labels.merge(metadata, how='left')

sample = pd.read_csv("input/train_features/S0001.csv", dtype={'time':np.float64, 'mass':np.float64, 'intensity':np.int64})
mass = np.round(sample.mass).values.astype(int)
intensity = sample.intensity.values
min_time = 0
max_time = 49
time_step = 1

# temp_step = 6
max_time_id = (max_time - min_time) // time_step

# temp_query_range = 10
time_query_range = 2
prob_smooth=6

res = np.zeros((535+1, max_time), dtype=np.float32) - 1
time = sample.time.values
step_pos = np.where(np.diff(sample['mass'].values, prepend=0) < 0)[0]
step_time = [np.mean(v) for v in np.split(time, step_pos)][1:-1]
time_bands = [[] for t in range(max_time + time_query_range + 1)]  # temp: list of steps
import math
for step, t in enumerate(step_time):
    t = math.floor(t)
    if 0 <= t < len(time_bands):
        time_bands[t].append(step)

for temp_id in range(max_time_id):
    t = min_time + temp_id * time_step
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
        res[mass[i], temp_id] = intensity[i]

res.shape
targetcols = ['aromatic', 'hydrocarbon', 'carboxylic_acid',
       'nitrogen_bearing_compound', 'chlorine_bearing_compound',
       'sulfur_bearing_compound', 'alcohol', 'other_oxygen_bearing_compound',
       'mineral']

print(train.shape, val.shape, test.shape)
train['intensity'] = train['intensity'].replace(0, np.nan)
val['intensity'] = val['intensity'].replace(0, np.nan)
test['intensity'] = test['intensity'].replace(0, np.nan)
train_noise, val_noise, test_noise = [], [], []
train_noise.append(train.groupby(['sample_number','mass'], dropna=True)['intensity'].min())
val_noise.append(val.groupby(['sample_number','mass'], dropna=True)['intensity'].min())
test_noise.append(test.groupby(['sample_number','mass'], dropna=True)['intensity'].min())
for q in tqdm([0.01,2,5,20]):
    train_noise.append(train.groupby(['sample_number', 'mass'], dropna=True)['intensity'].quantile(q/100))
    val_noise.append(val.groupby(['sample_number', 'mass'], dropna=True)['intensity'].quantile(q/100))
    test_noise.append(test.groupby(['sample_number', 'mass'], dropna=True)['intensity'].quantile(q/100))

def cnn():
    inp = tf.keras.layers.Input(shape=(TIMESTEPS, 237))

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
    conv_out = tf.keras.layers.Conv1D(16, 1, padding='same')(c)

    x = tf.keras.layers.Flatten()(c)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)

    out = tf.keras.layers.Dense(len(targetcols), activation='sigmoid')(x)
    model = tf.keras.models.Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

def get_timespan(df):
    x = df.groupby(['sample_number', 'mass', 'time_id'])['intensity'].mean()
    print(x.shape)
    x = x.values.reshape((-1, TIMESTEPS, 237))
    return x

def train_generator(df, noise_l, targets, sample_ids, batch_size=16, scale=1):
    while 1:
        keep_idx = np.random.permutation(sample_ids)[:batch_size]
        df_tmp = df[df.sample_number.isin(keep_idx)]
        noise_tmp = []
        for noise in noise_l:
            noise_tmp.append(noise.index.get_level_values(0).isin(keep_idx))
        y_tmp = targets[targets.sample_number.isin(keep_idx)]
        scale = np.random.normal(loc=0.0, scale=1.0, size=None)
        percentile = np.random.randint(4) + 1
        yield create_dataset_part(df_tmp, noise_tmp, y_tmp[targetcols].values, scale=scale, percentile=percentile)
        gc.collect()


def create_dataset(df, df_min, y):
    return create_dataset_part(df, df_min, y[targetcols].values, scale=1, percentile=0, subsample=1)

def create_dataset_part(df, df_noise, y, scale=1, percentile=0, subsample=0.8):
    if(subsample < 1):
        trn_ind, _ = train_test_split(df.index, train_size=subsample)
    else:
        trn_ind = df.index
    x = get_timespan(df)
    x = x - df_noise[0].values.reshape((-1, 1, 237))
    if (percentile > 0):
        x = np.clip(x - df_noise[percentile].values.reshape((-1, 1, 237)), 0, 1e9)

    x = np.nan_to_num(x)

    sum_vals = np.sum(x, axis=2)
    sum_vals[sum_vals == 0] = 1
    x = x / sum_vals.reshape((-1, TIMESTEPS, 1))

    x = scale*x

    return(x, y)

def scheduler(epoch, lr):
    if epoch < 6:
        return 0.001
    elif epoch < 9:
        return 0.0001
    else:
        return 0.00001

Xtrain, Ytrain = create_dataset(train, train_min, labels)
Xval, _ = create_dataset(val, val_min, labels)
Xtest, _ = create_dataset(test, test_min, labels)


kfold = KFold(n_splits=config.n_fold, random_state=config.seed, shuffle=True)
oof = np.zeros((len(labels), len(targetcols)))
val_pred = []
test_pred = []
next(train_set)
train.groupby("sample_number")['mass'].nunique().unique()
train.groupby("sample_number")['time_id'].nunique().unique()

train.fillna(0, inplace=True)
# Iterate through each fold
scores = []
16 * 49 * 237
for fold, (trn_ind, val_ind) in enumerate(kfold.split(labels, labels)):
    train_samples = labels.sample_number.values[trn_ind]
    train_set = train_generator(train, train_noise, labels, train_samples, batch_size=16)
    #x_train, y_train = Xtrain[trn_ind], Ytrain[trn_ind]
    x_val, y_val = Xtrain[val_ind], Ytrain[val_ind]
    model = cnn()
    callback = [tf.keras.callbacks.EarlyStopping(patience=5),
                tf.keras.callbacks.ModelCheckpoint(f'cnn_{fold}.h5', save_best_only=True, save_weights_only=True),
                tf.keras.callbacks.ReduceLROnPlateau(patience=3)]
    model.fit(train_set, steps_per_epoch=50, epochs=100, verbose=1, validation_data=(x_val, y_val), callbacks=callback)
    model.load_weights(f"cnn_{fold}.h5")
    oof[val_ind] = model.predict(x_val)

    val_pred.append(model.predict(Xval, verbose=0))
    test_pred.append(model.predict(Xtest, verbose=0))

y = labels[targetcols].values
score = np.round(aggregated_log_loss(y, oof), 3)
print(f"CV Score: {aggregated_log_loss(y, oof)}")
#0.15585355122164415 - 0.2087
# CV Score: 0.1546591230320039

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