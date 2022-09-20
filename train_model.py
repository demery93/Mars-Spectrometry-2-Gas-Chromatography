import pandas as pd
import numpy as np
import gc
from config import config
import model_definitions
import tensorflow as tf
import tensorflow_addons as tfa
from dataset import MarsSpectrometryDataset
from utils import aggregated_log_loss, CosineAnnealingWarmRestarts

nb_epochs = 35
optimizers = 'adamW'
scheduler = 'CosineAnnealingWarmRestarts'
scheduler_period = 16
scheduler_t_mult =  1.41421

initial_lr = 0.0005
save_period = 10
grad_clip = 64
labels_smooth = 0.001


labels = pd.read_csv("input/train_labels.csv")
sub = pd.read_csv("input/submission_format.csv")
labels.set_index('sample_id', inplace=True)
oof = labels.copy()

experiment = {"time_step":1,
              "max_time": 50,
              "timesteps":50,
              "max_mass":250,
              "min_mass":1,
              "nions":249,
              "model":'cnn1d',
              'fold':0}

optimizer = tfa.optimizers.AdamW(learning_rate=initial_lr, weight_decay=0.0001)
cls = model_definitions.__dict__[experiment['model']](experiment['timesteps'], experiment['nions'])
cls.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=labels_smooth))
scheduler = CosineAnnealingWarmRestarts(initial_learning_rate=initial_lr, first_decay_steps=scheduler_period, t_mul=scheduler_t_mult)

callbacks = [tf.keras.callbacks.LearningRateScheduler(scheduler),
             tf.keras.callbacks.ModelCheckpoint(
                filepath='model_paths',
                save_weights_only=True,
                save_best_only=False,
                monitor='val_loss',
                save_freq=20)]
train_ds = MarsSpectrometryDataset(
    fold=experiment['fold'],
    is_training=True,
    dataset_type='train',
    time_step=experiment['time_step'],
    max_time=experiment['max_time'],
    max_mass=experiment['max_mass'],
    min_mass=experiment['min_mass'])

val_ds = MarsSpectrometryDataset(
    fold=experiment['fold'],
    is_training=False,
    dataset_type='train',
    time_step=experiment['time_step'],
    max_time=experiment['max_time'],
    max_mass=experiment['max_mass'],
    min_mass=experiment['min_mass'])

history = cls.fit(
    train_ds,
    verbose=1,
    epochs=nb_epochs,
    batch_size=8,
    validation_data=val_ds,
    callbacks=callbacks
)
val_pred = np.zeros((len(val_ds.sample_ids), 9))
for i in range(config.tta):
    val_pred += cls.predict(val_ds) / config.tta

validation = labels[labels.index.isin(val_ds.sample_ids)]
validation[validation.columns] = val_pred

y = labels[labels.index.isin(val_ds.sample_ids)].values
print(aggregated_log_loss(y, validation.values))

test_ds = MarsSpectrometryDataset(
    fold=0,
    is_training=False,
    dataset_type='test_val',
    time_step=experiment['time_step'],
    max_time=experiment['max_time'],
    max_mass=experiment['max_mass'],
    min_mass=experiment['min_mass'])

preds = np.zeros(sub[config.targetcols].shape)
for j in range(config.tta):
    preds += cls.predict(test_ds) / config.tta

sub[config.targetcols] = preds
validation.to_csv(f"validation/validation_{experiment['model']}_fold{experiment['fold']}.csv", index=False, header=True)
sub.to_csv(f"output/submission_{experiment['model']})fold{experiment['fold']}.csv", index=False, header=True)