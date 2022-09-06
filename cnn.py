import pandas as pd
import numpy as np
import gc
import einops
from sklearn.model_selection import KFold
from config import config
import tensorflow as tf
from sklearn.metrics import log_loss

train = pd.read_feather("processed/train.f")
val = pd.read_feather("processed/val.f")
test = pd.read_feather("processed/test.f")
labels = pd.read_csv("input/train_labels.csv")
metadata = pd.read_feather("processed/metadata.f")
sub = pd.read_csv("input/submission_format.csv")
labels = labels.merge(metadata, how='left')

targetcols = ['aromatic', 'hydrocarbon', 'carboxylic_acid',
       'nitrogen_bearing_compound', 'chlorine_bearing_compound',
       'sulfur_bearing_compound', 'alcohol', 'other_oxygen_bearing_compound',
       'mineral']

train_min = pd.pivot_table(train, index=['sample_number','mass'], columns='time_id', values='intensity', aggfunc='min')
val_min = pd.pivot_table(val, index=['sample_number','mass'], columns='time_id', values='intensity', aggfunc='min')
test_min = pd.pivot_table(test, index=['sample_number','mass'], columns='time_id', values='intensity', aggfunc='min')

train = pd.pivot_table(train, index=['sample_number','mass'], columns='time_id', values='intensity', aggfunc='mean')
val = pd.pivot_table(val, index=['sample_number','mass'], columns='time_id', values='intensity', aggfunc='mean')
test = pd.pivot_table(test, index=['sample_number','mass'], columns='time_id', values='intensity', aggfunc='mean')

train = train - train_min
val = val - val_min
test = test - test_min


TIMESTEPS = train.shape[1]
gc.collect()

test = test[train.columns]
val = val[train.columns]

del train_min, test_min, val_min
gc.collect()

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

Xtrain, Ytrain = create_dataset(train, labels, scale=2)
Xval, _ = create_dataset(val, labels, scale=2)
Xtest, _ = create_dataset(test, labels, scale=2)


kfold = KFold(n_splits=config.n_fold, random_state=config.seed, shuffle=True)
oof = np.zeros((len(labels), len(targetcols)))
val_pred = np.zeros((len(Xval), len(targetcols)))
test_pred = np.zeros((len(Xtest), len(targetcols)))

# Iterate through each fold
scores = []
for fold, (trn_ind, val_ind) in enumerate(kfold.split(labels, labels)):
    x_train, y_train = Xtrain[trn_ind], Ytrain[trn_ind]
    x_val, y_val = Xtrain[val_ind], Ytrain[val_ind]
    model = cnn()
    callback = [tf.keras.callbacks.EarlyStopping(patience=7),
                tf.keras.callbacks.ModelCheckpoint('cnn.h5', save_best_only=True, save_weights_only=True),
                tf.keras.callbacks.ReduceLROnPlateau(patience=4)]
    model.fit(x_train, y_train, batch_size=16, epochs=100, verbose=1, validation_data=(x_val, y_val), callbacks=callback)
    model.load_weights("cnn.h5")
    oof[val_ind] = model.predict(x_val)

    val_pred += model.predict(Xval) / 5
    test_pred += model.predict(Xtest) / 5

y = labels[targetcols].values
scores = []
for i in range(len(targetcols)):
    score = log_loss(y[:,i], oof[:,i], eps=1e-7)
    scores.append(score)

score = np.round(np.mean(scores), 3)
print(f"CV Score: {np.mean(scores)}")
#CV Score: 0.140840184483223
sub = pd.read_csv("input/submission_format.csv")

pred = np.concatenate([val_pred, test_pred], axis=0)
pred = pd.DataFrame(pred, columns=targetcols)
for c in targetcols:
    sub[c] = pred[c]

sub.to_csv(f"output/submission_{score}.csv", index=False, header=True)
for i in range(len(targetcols)):
    print(oof[:,i].mean(), sub[targetcols].values[:,i].mean())

targetcols
sub.columns