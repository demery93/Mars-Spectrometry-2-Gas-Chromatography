import pandas as pd
import numpy as np

import os
import gc

from tqdm import tqdm
from config import config
from utils import fill_t_bins
import multiprocessing
from sklearn.model_selection import KFold, StratifiedKFold

def preprocess(src, dst, is_derivatized):
    '''

    This function iterates through the data directories, processes each sample, and stores it in an intensity
    dataframe and a timeerature dataframe
    '''
    sample = pd.read_csv(src, dtype={'time':np.float64, 'mass':np.float64, 'intensity':np.int64})
    sample = sample.loc[(sample['mass'] % 1 < 0.3) | (sample['mass'] % 1 > 0.7)].reset_index(drop=True)
    sample['mass'] = sample['mass'].round().astype(int)
    sample = sample[sample.mass < config.max_mass].reset_index(drop=True)
    sample = sample[sample.mass > 0].reset_index(drop=True)
    sample.loc[sample.mass == 4,'intensity'] = 0
    res = []

    for m in sorted(sample["mass"].unique()):
        t = sample[sample["mass"] == m]["time"].values
        intensity = sample[sample["mass"] == m]["intensity"].values

        t_bins, i_bins = fill_t_bins(t, intensity)
        if len(t_bins) < 4:
            print(f'Skip m {m} i {len(intensity)} t_bins {len(t_bins)} {src} is derivatized: {is_derivatized}')
            continue

        nan_values = ~np.isfinite(i_bins)
        if nan_values.sum() > 0:
            print(f'{src} nan: {nan_values.sum()}')
            t_bins = t_bins[~nan_values]
            i_bins = i_bins[~nan_values]

            if len(i_bins) < 4:
                print(f'No non null values: count {len(i_bins)} m {m} {src}  is derivatized: {is_derivatized}')
                continue

        if is_derivatized:
            t_cur = t
            i_cur = intensity
        else:
            t_cur = t_bins.copy()
            i_cur = i_bins.copy()

        i_sub_min = i_cur - np.min(i_cur)
        i_sub_q5 = i_cur - np.quantile(i_cur, 0.05)
        i_sub_q10 = i_cur - np.quantile(i_cur, 0.10)
        i_sub_q20 = i_cur - np.quantile(i_cur, 0.20)


        for i, ti in enumerate(t_cur):
            res.append({
                'time': ti,
                'mass': m,
                'intensity': i_cur[i],
                'intensity_sub_min': i_sub_min[i],
                'intensity_sub_q5': i_sub_q5[i],
                'intensity_sub_q10': i_sub_q10[i],
                'intensity_sub_q20': i_sub_q20[i],
            })
            # res.append([ti, m, a_cur[i], a_no_bg[i], a_sub_min[i], a_sub_q5[i], a_sub_q10[i], a_sub_q20[i]])

    res_df = pd.DataFrame(
        res
    )

    res_df = res_df.sort_values(['time', 'mass'], axis=0)
    res_df.reset_index(drop=True, inplace=True)
    res_df.to_feather(dst)


def preprocess_all_features():
    metadata = pd.read_csv("input/metadata.csv")

    requests = []
    dst_dir = config.processed_feature_path
    os.makedirs(dst_dir, exist_ok=True)

    pool = multiprocessing.Pool(16)

    for _, row in metadata.iterrows():
        src_dir = f'input/{row["split"]}_features'
        sample_id = row["sample_id"]
        is_derivatized = row["derivatized"] == 1
        requests.append([f'{src_dir}/{sample_id}.csv', f'{dst_dir}/{sample_id}.f', is_derivatized])
        #preprocess(f'{src_dir}/{sample_id}.csv', f'{dst_dir}/{sample_id}.csv')

    pool.starmap(preprocess, requests)

def split_to_folds():
    kf = StratifiedKFold(n_splits=config.n_folds, random_state=config.seed, shuffle=True)
    metadata = pd.read_csv("input/metadata.csv")

    metadata = metadata[metadata.split == 'train']
    metadata['fold'] = -1
    for fold, (train_index, test_index) in enumerate(kf.split(metadata.sample_id, metadata.derivatized.fillna(0))):
        print(fold, test_index)
        metadata.loc[test_index, 'fold'] = fold

    print(metadata['fold'].value_counts())
    metadata[['sample_id', 'split', 'fold', 'derivatized']].to_csv('processed/folds.csv', index=False)

if __name__ == "__main__":
    split_to_folds()
    preprocess_all_features()