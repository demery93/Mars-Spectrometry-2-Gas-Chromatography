import pandas as pd
import numpy as np
import gc
import einops
from sklearn.model_selection import KFold
from config import config
import tensorflow as tf
from utils import aggregated_log_loss
from einops.layers.keras import Rearrange

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

train_min = train.groupby(['sample_number','mass'])['intensity'].min()
#train_01 = train.groupby(['sample_number','mass'])['intensity'].quantile(0.01)
val_min = val.groupby(['sample_number','mass'])['intensity'].min()
test_min = test.groupby(['sample_number','mass'])['intensity'].min()

ionlist = [i for i in range(13, 250)]  # max number of ions for all models is 250
fill = pd.DataFrame({'mass': ionlist})
fills = []
for i in range(TIMESTEPS):
    for sample in labels.sample_number.unique():
        tmp = fill.copy()
        tmp['time_id'] = i
        tmp['sample_number'] = sample
        fills.append(tmp)
fill = pd.concat(fills, ignore_index=True)
fill = fill.set_index(['sample_number','mass','time_id'])

del fills, tmp
gc.collect()
sample = train.sample_number.unique()[:16]
tmp = train[train.sample_number.isin(sample)]
tmp2 = tmp.groupby(['sample_number','mass','time_id'])['intensity'].mean()
tmp3 = fill.join(tmp2, how='left').fillna(0)
tmp4 = tmp3.values.reshape((-1, TIMESTEPS, 237))

tmp.time_id.max()
len(tmp.mass.unique())
len(tmp.time_id.unique())
tmp3.values.reshape((-1, 49, 237))
7
train = pd.pivot_table(train, index=['sample_number','mass'], columns='time_id', values='intensity', aggfunc='mean')
val = pd.pivot_table(val, index=['sample_number','mass'], columns='time_id', values='intensity', aggfunc='mean')
test = pd.pivot_table(test, index=['sample_number','mass'], columns='time_id', values='intensity', aggfunc='mean')

TIMESTEPS = train.shape[1]
gc.collect()

test = test[train.columns]
val = val[train.columns]

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
    X = df.values
    return X

def train_generator(df, df_min, targets, sample_ids, batch_size=16, scale=1):
    while 1:
        keep_idx = np.random.permutation(sample_ids)[:batch_size]
        df_tmp = df[df.index.get_level_values(0).isin(keep_idx)]
        df_min_tmp = df_min[df_min.index.get_level_values(0).isin(keep_idx)]
        y_tmp = targets[targets.sample_number.isin(keep_idx)]
        scale = np.random.normal(loc=0.0, scale=1.0, size=None)
        percentile = np.random.permutation([0.1, 5, 10, 20])[0]
        yield create_dataset_part(df_tmp, df_min_tmp, y_tmp[targetcols].values, scale=scale, percentile=percentile)
        gc.collect()

def create_dataset(df, df_min, y):
    return create_dataset_part(df, df_min, y[targetcols].values, scale=1)

def create_dataset_part(df, df_min, y, scale=1, percentile=0):
    df = df - df_min.values.reshape((-1,1))

    x = get_timespan(df)
    if (percentile > 0):
        xp = np.nanpercentile(x, percentile, axis=1)
        x = np.clip(x - xp.reshape((-1,1)), 0, 1e9)

    x = np.nan_to_num(x)
    x = einops.rearrange(x, '(b i) t -> b t i', i=237)

    sum_vals = np.sum(np.nan_to_num(x), axis=2)
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

# Iterate through each fold
scores = []
for fold, (trn_ind, val_ind) in enumerate(kfold.split(labels, labels)):
    train_samples = labels.sample_number.values[trn_ind]
    train_set = train_generator(train, train_min, labels, train_samples, batch_size=32)
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