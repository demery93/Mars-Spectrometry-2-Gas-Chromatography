import pandas as pd
import os

from utils import aggregated_log_loss, load_config_data
from config import config

cfg = load_config_data("200_ensemble")
val_labels = pd.read_csv("input/val_labels.csv")

model_params = cfg['model_params']
experiment_params = cfg['experiment_params']
folds = pd.read_csv("processed/folds.csv")
labels = pd.read_csv("input/train_labels.csv", index_col=['sample_id'])
sub = pd.read_csv("input/submission_format.csv", index_col=['sample_id'])
oof = labels.copy()
for c in sub.columns:
    sub[c] = 0
    oof[c] = 0

for model in experiment_params['models']:
    sub_model = sub.copy()
    oof_model = oof.copy()
    for c in sub_model.columns:
        oof_model[c] = 0
        sub_model[c] = 0
    for i in range(config.n_folds):
        train_samples = folds[folds.fold==i]['sample_id'].values
        oof_dir = f"{config.output_path}/oof/{model}_{i}"
        sub_dir = f"{config.output_path}/sub/{model}_{i}"
        tmp_oof = pd.read_csv(f"{oof_dir}/{model}_fold{i}.csv")
        tmp_sub = pd.read_csv(f"{sub_dir}/{model}_fold{i}.csv", index_col=['sample_id'])
        oof_model[oof_model.index.isin(train_samples)] = tmp_oof.values
        sub_model += tmp_sub.values / config.n_folds


    print(f"CV Score for Model {model}: {aggregated_log_loss(labels.values, oof_model.values)}")  # 0.1802036797157865
    sub += sub_model.values / len(experiment_params['models'])
    oof += oof_model.values / len(experiment_params['models'])
    val_pred = val_labels[['sample_id']].merge(sub_model.reset_index())

print(f"CV Score for Model {model_params['model_cls']}: {aggregated_log_loss(labels.values, oof.values)}")  # 0.1665458760452786

os.makedirs(f"{config.output_path}/sub/{model_params['model_name']}", exist_ok=True)
sub.reset_index().to_csv(f"{config.output_path}/sub/{model_params['model_name']}/submission.csv", index=False, header=True)


aggregated_log_loss(labels.values, oof.values, verbose=True)
val_pred = val_labels[['sample_id']].merge(sub.reset_index())
y = val_labels.values[:, 1:].astype(float)
pred = val_pred.values[:, 1:]
print(aggregated_log_loss(y, pred)) #0.17227686768787626
