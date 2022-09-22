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

fold = int(sys.argv[1])
experiment_name = sys.argv[2]

cfg = load_config_data(experiment_name)

model_params = cfg['model_params']
dataset_params = cfg['dataset_params']
train_params = cfg['train_params']

labels = pd.read_csv("input/train_labels.csv")
sub = pd.read_csv("input/submission_format.csv")
labels.set_index('sample_id', inplace=True)
oof = labels.copy()

checkpoints_dir = f"{config.output_path}/checkpoints/{model_params['model_cls']}_{fold}"
#tensorboard_dir = f"{config.output_path}/tensorboard/{experiment['model']}_{experiment['fold']}"
#logger = tf.summary.SummaryWriter()
oof_dir = f"{config.output_path}/oof/{model_params['model_cls']}_{fold}"
os.makedirs(checkpoints_dir, exist_ok=True)
#os.makedirs(tensorboard_dir, exist_ok=True)
os.makedirs(oof_dir, exist_ok=True)
optimizer = tfa.optimizers.AdamW(learning_rate=train_params['initial_lr'], weight_decay=0.0001)
cls = model_definitions.__dict__[model_params['model_cls']](dataset_params['timesteps'], dataset_params['nions'])
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
    time_step=dataset_params['time_step'],
    max_time=dataset_params['max_time'],
    max_mass=dataset_params['max_mass'],
    min_mass=dataset_params['min_mass'])

val_ds = MarsSpectrometryDataset(
    fold=fold,
    is_training=False,
    dataset_type='train',
    time_step=dataset_params['time_step'],
    max_time=dataset_params['max_time'],
    max_mass=dataset_params['max_mass'],
    min_mass=dataset_params['min_mass'])

history = cls.fit(
    train_ds,
    verbose=1,
    epochs=train_params['nb_epochs'],
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
    time_step=dataset_params['time_step'],
    max_time=dataset_params['max_time'],
    max_mass=dataset_params['max_mass'],
    min_mass=dataset_params['min_mass'])

preds = np.zeros(sub[config.targetcols].shape)
for j in range(config.tta):
    preds += cls.predict(test_ds) / config.tta

sub[config.targetcols] = preds
validation.to_csv(f"validation/validation_{model_params['model_cls']}_fold{fold}.csv", index=False, header=True)
sub.to_csv(f"output/submission_{model_params['model_cls']}_fold{fold}.csv", index=False, header=True)