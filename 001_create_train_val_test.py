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

    ## For each sample, we need to ensure that mass / time id combination has at least 1 record
    size = 49 // config.sample_rate
    ionlist = [i for i in range(13,250)] * size  # max number of ions for all models is 250
    times = [i % size for i in range(len(ionlist))]
    fill = pd.DataFrame({'mass': ionlist, 'time_id':times})

    sample_number = 0
    for dir in dirs:
        for file in tqdm(os.listdir(dir)):
            sample = pd.read_csv(f"{dir}/{file}", dtype={'time':np.float64, 'mass':np.float64, 'intensity':np.int64})
            sample['sample_number'] = sample_number
            sample['sample_number'] = sample['sample_number']
            sample['mass'] = np.round(sample['mass']).astype(np.int16)
            sample = sample[(sample.mass > 12) & (sample.mass < 250)].reset_index(drop=True)

            ## Create time id
            sample['time_id'] = (sample['time'] // config.sample_rate).astype(int)
            sample.drop(['time'], axis=1, inplace=True)


            # Make sure all ions are present in all samples
            ionfill = fill[~(fill['mass'].astype(str) + fill['time_id'].astype(str)).isin((sample['mass'].astype(str) + sample['time_id'].astype(str)))].copy()
            ionfill['sample_number'] = sample_number
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

    metadata = pd.read_csv("input/metadata.csv")
    metadata = metadata.merge(lkp, on=['sample_id'], how='left')

    df = df.merge(metadata[['sample_number','split']], how='left', on='sample_number')

    train = df[df.split == 'train'].reset_index(drop=True)
    val = df[df.split == "val"].reset_index(drop=True)
    test = df[df.split == "test"].reset_index(drop=True)

    del df
    gc.collect()

    train.drop(['split'], axis=1, inplace=True)
    val.drop(['split'], axis=1, inplace=True)
    test.drop(['split'], axis=1, inplace=True)

    print(f"Number of train samples: {len(train['sample_number'].unique())}")
    print(f"Number of val samples: {len(val['sample_number'].unique())}")
    print(f"Number of test samples: {len(test['sample_number'].unique())}")

    train.to_feather("processed/train.f")
    val.to_feather("processed/val.f")
    test.to_feather("processed/test.f")
    metadata.to_feather("processed/metadata.f")

if __name__ == "__main__":
    main()
