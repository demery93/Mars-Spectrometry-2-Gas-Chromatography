import os

class ParamConfig:
    def __init__(self):
        #Number of runs per model
        self.n_folds = 5
        self.sample_rate = 1
        self.n_fold = 5
        self.seed = 42
        self.max_mass = 250
        self.min_mass = 13
        self.min_time = 0
        self.max_time = 50

        self.train_data_path = 'input/train_features/'
        self.val_data_path = 'input/val_features/'
        self.test_data_path = 'input/test_features/'
        self.processed_feature_path = 'processed/features'
        self.train_label_path = 'input/train_labels.csv'
        self.targetcols = ['aromatic', 'hydrocarbon', 'carboxylic_acid','nitrogen_bearing_compound',
                           'chlorine_bearing_compound','sulfur_bearing_compound', 'alcohol', 'other_oxygen_bearing_compound','mineral']



## initialize a param config
config = ParamConfig()