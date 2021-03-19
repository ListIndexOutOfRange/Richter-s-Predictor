import warnings
warnings.filterwarnings('ignore')

import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from utils import load


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



# +-------------------------------------------------------------------------------------------+ #
# |              Finding relation between geo_levels using seq_to_seq method (LSTM)           | #
# +-------------------------------------------------------------------------------------------+ #

def encode_geolevels_one_hot_inputs(X, test):
    l1 = pd.concat([X['geo_level_1_id'], test['geo_level_1_id']])
    l2 = pd.concat([X['geo_level_2_id'], test['geo_level_2_id']])
    inputs = pd.concat([l1, l2]).to_numpy().reshape(-1,1)
    l1 = l1.to_numpy().reshape(-1,1)
    l2 = l2.to_numpy().reshape(-1,1)
    onehot = OneHotEncoder(sparse=False)
    onehot.fit(inputs)
    return np.stack((onehot.transform(l1), onehot.transform(l2)), axis=1)
    #return 'TESTING', 'TESTING'


def encode_geolevels_one_hot(X, test):
    l1 = pd.concat([X['geo_level_1_id'], test['geo_level_1_id']])
    l2 = pd.concat([X['geo_level_2_id'], test['geo_level_2_id']])
    l3 = pd.concat([X['geo_level_3_id'], test['geo_level_3_id']])
    inputs = pd.concat([l1, l2]).to_numpy().reshape(-1,1)
    l1 = l1.to_numpy().reshape(-1,1)
    l2 = l2.to_numpy().reshape(-1,1)
    onehot = OneHotEncoder(sparse=False)
    onehot.fit(inputs)
    inputs = np.stack((onehot.transform(l1), onehot.transform(l2)), axis=1)
    l3_hot = np.array(pd.get_dummies(l3))
    return inputs, l3_hot




# +-------------------------------------------------------------------------------------------+ #
# |                                         Run Preprocessing                                 | #
# +-------------------------------------------------------------------------------------------+ #

def preprocess(rootdir: str) -> None:
    X, y, test = load(rootdir)
    for i in range(3):
        feature = f'geo_level_{i+1}_id'  
        X = add_conditional_probabilites(X, y, feature)
    X.to_csv('train_values_preprocessed.csv')


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("rootdir", help="path to the folder containing the csv files")
	return parser.parse_args()


def test(rootdir):
    X, y, test = load(rootdir)
    inputs = encode_geolevels_one_hot_inputs(X, test)
    print(inputs[:7])
    #print(l3[:7])

if __name__ == '__main__': test(parse_args().rootdir)