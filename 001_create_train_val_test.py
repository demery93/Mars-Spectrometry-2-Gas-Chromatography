import pandas as pd
import numpy as np

import os
import gc

from tqdm import tqdm
from config import config

def read_and_concatenate():
    '''
    :return: stacked dataframe of all samples

    This function iterates through the data directories, processes each sample, and stores it in an abundance
    dataframe and a temperature dataframe
    '''
    dirs = [config.train_data_path, config.val_data_path, config.test_data_path]
    sample_numbers = []
    sample_ids = []
    samples = []

    ionlist = [i for i in range(13,250)]  # max number of ions for all models is 250
    fill = pd.DataFrame({'mass': ionlist})

    sample_number = 0
    for dir in dirs:
        for file in tqdm(os.listdir(dir)):
            sample = pd.read_csv(f"{dir}/{file}", dtype={'time':np.float64, 'mass':np.float64, 'intensity':np.int64})
            sample['sample_number'] = sample_number
            sample['sample_number'] = sample['sample_number']
            sample['mass'] = np.round(sample['mass']).astype(np.int16)
            sample = sample[(sample.mass > 12) & (sample.mass < 250)].reset_index(drop=True)
            t = sample.time.min()

            # Make sure all ions are present in all samples
            ionfill = fill[~fill['mass'].isin(sample['mass'])].copy()
            ionfill['sample_number'] = sample_number
            ionfill['time'] = t
            sample = pd.concat([sample, ionfill], axis=0, ignore_index=True).fillna(0)

            samples.append(sample)
            sample_numbers.append(sample_number)
            sample_ids.append(file.split('.')[0])

            sample_number += 1 #Increment sample number

    df = pd.concat(samples, axis=0)
    lkp = pd.DataFrame({"sample_id":sample_ids, "sample_number":sample_numbers})

    return df, lkp


def main():
    df, lkp = read_and_concatenate()

    df['time_id'] = (df['time'] // config.sample_rate).astype(int)
    df.drop(['time'], axis=1, inplace=True)

    metadata = pd.read_csv("input/metadata.csv")
    metadata = metadata.merge(lkp, on=['sample_id'], how='left')

    train = df[df['sample_number'].isin(metadata[metadata.split=='train'].sample_number)].reset_index(drop=True)
    val = df[df['sample_number'].isin(metadata[metadata.split=='val'].sample_number)].reset_index(drop=True)
    test = df[df['sample_number'].isin(metadata[metadata.split=='test'].sample_number)].reset_index(drop=True)

    del df
    gc.collect()

    print(f"Number of train samples: {len(train['sample_number'].unique())}")
    print(f"Number of val samples: {len(val['sample_number'].unique())}")
    print(f"Number of test samples: {len(test['sample_number'].unique())}")

    train.to_feather("processed/train.f")
    val.to_feather("processed/val.f")
    test.to_feather("processed/test.f")
    metadata.to_feather("processed/metadata.f")

if __name__ == "__main__":
    main()
