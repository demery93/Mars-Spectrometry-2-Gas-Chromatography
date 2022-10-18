import pandas as pd
import numpy as np
import gc
from config import config
import model_definitions
import tensorflow as tf
import tensorflow_addons as tfa
from dataset import MarsSpectrometryDataset
from utils import aggregated_log_loss, CosineAnnealingWarmRestarts, load_config_data
import os
import sys

fold = 0
experiment_name = "100_conv1d"
cfg = load_config_data(experiment_name)

model_params = cfg['model_params']
dataset_params = cfg['dataset_params']
train_params = cfg['train_params']
predict_params = cfg['predict_params']

timesteps = int(dataset_params['max_time'] // dataset_params['time_step'])
nions = int(dataset_params['max_mass'] - dataset_params['min_mass'])

labels = pd.read_csv("input/train_labels.csv")
sub = pd.read_csv("input/submission_format.csv")
labels.set_index('sample_id', inplace=True)
oof = labels.copy()

checkpoints_dir = f"{config.output_path}/checkpoints/{model_params['model_name']}_{fold}/"
#tensorboard_dir = f"{config.output_path}/tensorboard/{experiment['model']}_{experiment['fold']}"
#logger = tf.summary.SummaryWriter()
oof_dir = f"{config.output_path}/oof/{model_params['model_name']}_{fold}"
sub_dir = f"{config.output_path}/sub/{model_params['model_name']}_{fold}"
os.makedirs(checkpoints_dir, exist_ok=True)
#os.makedirs(tensorboard_dir, exist_ok=True)
os.makedirs(oof_dir, exist_ok=True)
os.makedirs(sub_dir, exist_ok=True)
if(train_params['optimizer'] == 'adamW'):
    optimizer = tfa.optimizers.AdamW(learning_rate=train_params['initial_lr'], weight_decay=0.0001)
if(train_params['optimizer'] == 'sgd'):
    optimizer = tf.keras.optimizers.SGD(learning_rate=train_params['initial_lr'], momentum=0.9, nesterov=True, clipnorm=64)
cls = model_definitions.__dict__[model_params['model_cls']](timesteps, nions)
cls.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=train_params['labels_smooth']))
scheduler = CosineAnnealingWarmRestarts(initial_learning_rate=train_params['initial_lr'],
                                        first_decay_steps=train_params['scheduler_period'],
                                        t_mul=train_params['scheduler_t_mult'])

callbacks = [tf.keras.callbacks.LearningRateScheduler(scheduler),
             tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoints_dir,
                save_weights_only=True,
                save_best_only=False,
                monitor='val_loss',
                save_freq=20)]

train_ds = MarsSpectrometryDataset(
    fold=fold,
    is_training=True,
    dataset_type='train',
    batch_size=train_params['batch_size'],
    **dataset_params)

val_ds = MarsSpectrometryDataset(
    fold=fold,
    is_training=False,
    dataset_type='val',
    batch_size=predict_params['batch_size'],
    **dataset_params)


history = cls.fit(
    train_ds,
    verbose=1,
    epochs=train_params['nb_epochs'],
    validation_data=val_ds,
    callbacks=callbacks
)
val_pred = np.zeros((len(val_ds.sample_ids), 9))
for i in range(predict_params['tta']):
    val_pred += cls.predict(val_ds.load_val(), batch_size=predict_params['batch_size']) / predict_params['tta']

val_labels = pd.read_csv("input/val_labels.csv", index_col=['sample_id'])

y = val_labels.values
print(aggregated_log_loss(y, val_pred))
#0.1605

import pandas as pd
import os
dir = "processed/features/"
tmax = []
for f in os.listdir(dir):
    s = pd.read_feather(dir+f)
    tmax.append(s.time.max())

print(np.min(tmax))
print(np.max(tmax))