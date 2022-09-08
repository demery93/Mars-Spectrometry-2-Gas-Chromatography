import pandas as pd
import numpy as np

import os
import gc

from tqdm import tqdm
from config import config
from utils import fill_t_bins
import multiprocessing

def preprocess(src, dst):
    '''

    This function iterates through the data directories, processes each sample, and stores it in an intensity
    dataframe and a timeerature dataframe
    '''
    sample = pd.read_csv(src, dtype={'time':np.float64, 'mass':np.float64, 'intensity':np.int64})
    sample = sample.loc[(sample['mass'] % 1 < 0.3) | (sample['mass'] % 1 > 0.7)].reset_index(drop=True)
    sample['mass'] = sample['mass'].round().astype(int)
    sample = sample[(sample.mass >= config.min_mass) & (sample.mass < config.max_mass)].reset_index(drop=True)
    #max_int = np.nanmax(sample["intensity"]) # No Scaling in initial preprocessing
    res = []

    for m in sorted(sample["mass"].unique()):
        t = sample[sample["mass"] == m]["time"].values
        intensity = sample[sample["mass"] == m]["intensity"].values

        t_bins, i_bins = fill_t_bins(t, intensity)
        if len(t_bins) < 4:
            print(f'Skip m {m} i {len(intensity)} t_bins {len(t_bins)}')
            continue

        nan_values = ~np.isfinite(i_bins)
        if nan_values.sum() > 0:
            print(f'{src} nan: {nan_values.sum()}')
            t_bins = t_bins[~nan_values]
            a_bins = a_bins[~nan_values]

        bkg_2 = np.zeros_like(i_bins) + np.nanmin(i_bins)

        bkg_cur = np.interp(t, t_bins, bkg_2)
        t_cur = t
        i_cur = intensity

        i_no_bg = i_cur - bkg_cur
        i_no_bg_sub_min = i_no_bg - np.quantile(i_no_bg, 0.001)
        i_no_bg_sub_q5 = i_no_bg - np.quantile(i_no_bg, 0.05)
        i_no_bg_sub_q10 = i_no_bg - np.quantile(i_no_bg, 0.10)
        i_no_bg_sub_q20 = i_no_bg - np.quantile(i_no_bg, 0.20)

        i_sub_min = i_cur - np.quantile(i_cur, 0.001)
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

                'intensity_sub_bg': i_no_bg[i],

                'intensity_sub_bg_sub_min': i_no_bg_sub_min[i],
                'intensity_sub_bg_sub_q5': i_no_bg_sub_q5[i],
                'intensity_sub_bg_sub_q10': i_no_bg_sub_q10[i],
                'intensity_sub_bg_sub_q20': i_no_bg_sub_q20[i]
            })
            # res.append([ti, m, a_cur[i], a_no_bg[i], a_sub_min[i], a_sub_q5[i], a_sub_q10[i], a_sub_q20[i]])

    res_df = pd.DataFrame(
        res
    )

    res_df['mass'] = res_df['mass'].round().astype(int)

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
        requests.append([f'{src_dir}/{sample_id}.csv', f'{dst_dir}/{sample_id}.f'])
        #preprocess(f'{src_dir}/{sample_id}.csv', f'{dst_dir}/{sample_id}.csv')

    pool.starmap(preprocess, requests)

if __name__ == "__main__":
    preprocess_all_features()
