import warnings
warnings.filterwarnings('ignore')

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from typing import List
from utils import load, create_dir_if_needed


# +-------------------------------------------------------------------------------------------+ #
# |                             Conditional Probabilities of geo_levels                       | #
# +-------------------------------------------------------------------------------------------+ #

def add_conditional_probabilites(X, y, feature):
    features    = pd.merge(X, y, on='building_id')
    num_samples = {1: {}, 2: {}, 3: {}}
    damages     = {1: {}, 2: {}, 3: {}}
    probas      = {1: [], 2: [], 3: []}
    for i, j in tqdm((X[feature].value_counts()).iteritems()):
        for k in range(3):
            num_samples[k+1] = len(features[features['damage_grade']==k+1][features[feature]==i])
            damages[k+1][i]  = num_samples[k+1] / j
    for i in X[feature]:
        for j in range(3):
            probas[j+1].append(damages[j+1].get(i))
    for i in range(3):
        X[f'prob{i+1}_{feature}'] = probas[i+1]
    return X


def add_all_conditional_probabilites(X, y):
    for i in range(3):
        feature = f'geo_level_{i+1}_id'  
        X = add_conditional_probabilites(X, y, feature)
    return X




# +-------------------------------------------------------------------------------------------+ #
# |                                 One Hot Encoding of geo_levels                            | #
# +-------------------------------------------------------------------------------------------+ #

def concat_geolevels(X, test):
    level_1 = pd.concat([X['geo_level_1_id'], test['geo_level_1_id']])
    level_2 = pd.concat([X['geo_level_2_id'], test['geo_level_2_id']])
    level_3 = pd.concat([X['geo_level_3_id'], test['geo_level_3_id']])
    return level_1, level_2, level_3


def onehot_encode_geolevels_input_sequence(level_1, level_2):
    onehot = OneHotEncoder(sparse=False)
    onehot.fit(pd.concat([level_1,level_2]).to_numpy().reshape(-1, 1))
    level_1_onehot = onehot.transform(level_1.to_numpy().reshape(-1,1))
    level_2_onehot = onehot.transform(level_2.to_numpy().reshape(-1,1))
    return np.stack((level_1_onehot, level_2_onehot), axis=1)


def onehot_encode_geolevels(X, test):
    level_1, level_2, level_3 = concat_geolevels(X, test)
    geolevels_input_sequence  = onehot_encode_geolevels_input_sequence(level_1, level_2)
    level_3_onehot = np.array(pd.get_dummies(level_3))
    return geolevels_input_sequence, level_3_onehot




# +-------------------------------------------------------------------------------------------+ #
# |              Finding relation between geo_levels using seq_to_seq method (LSTM)           | #
# +-------------------------------------------------------------------------------------------+ #


# +-------------------------------------------------------------------------------------------+ #
# |                                         Run Preprocessing                                 | #
# +-------------------------------------------------------------------------------------------+ #

def preprocess(rootdir: str, steps = List[int]) -> None:
    output_dir = "preprocessing_outputs" 
    create_dir_if_needed(output_dir)
    X, y, test = load(rootdir)
    if 1 in steps:
        X = add_all_conditional_probabilites(X, y)
        X.to_csv(os.path.join(output_dir, 'train_values_preprocessed.csv'))
    if 2 in steps:
        inputs, level_3_onehot = onehot_encode_geolevels(X, test)
        lstm_data_dir = "lstm_data"
        create_dir_if_needed(os.path.join(output_dir, lstm_data_dir))
        for i in tqdm(range(inputs.shape[0])):
            filename = os.path.join(output_dir, lstm_data_dir, f"{i}.npz") 
            np.savez(filename, inputs=inputs[i], targets=level_3_onehot[i])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("rootdir", help="path to the folder containing the csv files")
    parser.add_argument('-s','--steps', nargs='+', help='<Required> Preprocessing steps',
                    required=True, type=int)
    return parser.parse_args()


def test(rootdir, steps):
    X, y, test = load(rootdir)
    inputs, level_3_onehot = onehot_encode_geolevels(X, test)
    print(inputs.shape)
    print(level_3_onehot.shape)
    print(40*'-')
    print(inputs[0].shape)
    print(level_3_onehot[0].shape)
    #print(os.listdir(rootdir))

if __name__ == '__main__':
    args = parse_args()
    preprocess(args.rootdir, args.steps)