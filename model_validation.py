import pandas as pd
import numpy as np
import gc
import einops
from sklearn.model_selection import KFold, train_test_split
from config import config
import tensorflow as tf
from sklearn.metrics import log_loss

train = pd.read_feather("processed/train.f")
labels = pd.read_csv("input/train_labels.csv")
metadata = pd.read_feather("processed/metadata.f")
labels = labels.merge(metadata, how='left')

targetcols = ['aromatic', 'hydrocarbon', 'carboxylic_acid',
       'nitrogen_bearing_compound', 'chlorine_bearing_compound',
       'sulfur_bearing_compound', 'alcohol', 'other_oxygen_bearing_compound',
       'mineral']

train_ind, val_ind = train_test_split(labels.sample_number, test_size=0.3, shuffle=True, random_state=42)

val = train[train.sample_number.isin(val_ind)].sort_values("sample_number").reset_index(drop=True)
train = train[train.sample_number.isin(train_ind)].sort_values("sample_number").reset_index(drop=True)

train_labels = labels.iloc[train_ind].sort_values("sample_number").reset_index(drop=True)
val_labels = labels.iloc[val_ind].sort_values("sample_number").reset_index(drop=True)

train_min = pd.pivot_table(train, index=['sample_number','mass'], columns='time_id', values='intensity', aggfunc='min')
val_min = pd.pivot_table(val, index=['sample_number','mass'], columns='time_id', values='intensity', aggfunc='min')

val = pd.pivot_table(val, index=['sample_number','mass'], columns='time_id', values='intensity', aggfunc='mean')
train = pd.pivot_table(train, index=['sample_number','mass'], columns='time_id', values='intensity', aggfunc='mean')

train = train - train_min.min(axis=1).values.reshape((-1,1))
val = val - val_min.min(axis=1).values.reshape((-1,1))

TIMESTEPS = train.shape[1]
gc.collect()

for c in train.columns:
    if c not in val.columns:
        val[c] = np.nan

def cnn():
    inp = tf.keras.layers.Input(shape=(TIMESTEPS, 237))

    c1 = tf.keras.layers.Conv1D(50, 2, strides=1, dilation_rate=2 ** 0, padding='same')(inp)
    c2 = tf.keras.layers.Conv1D(50, 2, strides=1, dilation_rate=2 ** 1, padding='same')(inp)
    c3 = tf.keras.layers.Conv1D(50, 2, strides=1, dilation_rate=2 ** 2, padding='same')(inp)
    c4 = tf.keras.layers.Conv1D(50, 2, strides=1, dilation_rate=2 ** 3, padding='same')(inp)
    c5 = tf.keras.layers.Conv1D(50, 2, strides=1, dilation_rate=2 ** 4, padding='same')(inp)
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
    X = np.nan_to_num(df.values)
    return X


def train_generator(df, targets, sample_ids, batch_size=16, scale=1):
    while 1:
        keep_idx = np.random.permutation(sample_ids)[:batch_size]
        df_tmp = df[df.index.get_level_values(0).isin(keep_idx)]
        y_tmp = targets[targets.sample_number.isin(keep_idx)]
        yield create_dataset_part(df_tmp, y_tmp[targetcols].values, 0, True, scale=scale)
        gc.collect()

def create_dataset(df, y, scale=2):
    return create_dataset_part(df, y[targetcols].values, scale=scale)

def create_dataset_part(df, y, noise_removal=None, percentile=20, scale=2, normalize='global', is_train=True):
    x = get_timespan(df)
    if(noise_removal=='min'):
        x = x - np.min(x, axis=1).reshape((-1,1))

    x = einops.rearrange(x, '(b i) t -> b t i', i=237)

    if(scale==1):
        max_vals = np.max(np.max(np.nan_to_num(x), axis=2), axis=1)
        max_vals[max_vals == 0] = 1
        x = x / max_vals.reshape((-1, 1, 1))
    if(scale==2):
        sum_vals = np.sum(np.nan_to_num(x), axis=2)
        sum_vals[sum_vals == 0] = 1
        x = x / sum_vals.reshape((-1, TIMESTEPS, 1))

    return(x, y)

def scheduler(epoch, lr):
    if epoch < 6:
        return 0.001
    elif epoch < 9:
        return 0.0001
    else:
        return 0.00001

xval = create_dataset(val, val_labels, scale=2)
kfold = KFold(n_splits=config.n_fold, random_state=config.seed, shuffle=True)
Xtrain, Ytrain = create_dataset(train, train_labels, scale=2)
oof = np.zeros((len(Ytrain), len(targetcols)))
val_pred = np.zeros((len(xval[0]), len(targetcols)))
# Iterate through each fold
scores = []
for fold, (trn_ind, val_ind) in enumerate(kfold.split(Xtrain, Ytrain)):
    x_train, y_train = Xtrain[trn_ind], Ytrain[trn_ind]
    x_val, y_val = Xtrain[val_ind], Ytrain[val_ind]
    model = cnn()
    callback = [tf.keras.callbacks.EarlyStopping(patience=5),
                tf.keras.callbacks.ModelCheckpoint('cnn.h5', save_best_only=True, save_weights_only=True),
                tf.keras.callbacks.ReduceLROnPlateau(patience=3)]
    #model.fit(train_set, steps_per_epoch=100, epochs=100, verbose=1, validation_data=val_set,callbacks=callback)
    model.fit(x_train, y_train, batch_size=8, epochs=100, verbose=1, validation_data=(x_val, y_val), callbacks=callback)
    model.load_weights("cnn.h5")
    holdout_pred = model.predict(x_val)
    oof[val_ind] = holdout_pred
    val_pred += model.predict(xval[0]) / 5

    scores = []
    for i in range(len(targetcols)):
        try:
            score = log_loss(y_val[:,i], holdout_pred[:,i], eps=1e-7)
            scores.append(score)
        except:
            pass

    val_score = np.round(np.mean(scores),3)
    print(f"Mean Aggregated Logloss for fold {fold+1}: {val_score}")

y = train_labels[targetcols].values
scores = []
for i in range(len(targetcols)):
    score = log_loss(y[:,i], oof[:,i], eps=1e-7)
    scores.append(score)

score = np.round(np.mean(scores), 3)
print(f"CV Score: {np.mean(scores)}")
np.mean(scores[:4])
#CV Score: 0.15091819688979083

y_val2 = val_labels[targetcols].values
scores = []
for i in range(len(targetcols)):
    score = log_loss(y_val2[:,i], val_pred[:,i], eps=1e-7)
    scores.append(score)

score = np.round(np.mean(scores), 3)
print(f"Holdout Score: {np.mean(scores)}")

'''
CV Score: 0.15275252103104803
Holdout Score: 0.15331449266665523
'''

