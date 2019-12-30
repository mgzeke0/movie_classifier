import argparse
import os
import pickle

import pandas as pd

from sklearn.model_selection import train_test_split

import tensorflow as tf

from config import DATASET_PATH, GENRES, RANDOM_SEED, TRAINED_MODEL_PATH, HYPERPARAMS
from train.train_functions import one_hot, load_batches, random_search, train_one_model, f1_score

tf.get_logger().setLevel('ERROR')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--randomsearch', help='Perform random search', action='store_true')
    parser.add_argument('--save', help='Save the model', action='store_true')

    args = parser.parse_args()
    data = pd.read_csv(os.path.join(DATASET_PATH, 'movies.csv'))
    data['genres_list'] = data['genres_list'].apply(eval)
    labels, reverse_one_hot_dict = one_hot(data['genres_list'], len(GENRES), GENRES.keys())

    pickle.dump(reverse_one_hot_dict, open(TRAINED_MODEL_PATH + 'reverse_dict.pkl', 'wb'))

    # Split data in train dev and test
    train_data, val_data, train_labels, val_labels = train_test_split(load_batches(data.id), labels, test_size=0.1, random_state=RANDOM_SEED)
    train_data, test_data, train_labels, test_labels = train_test_split(train_data, train_labels, test_size=0.2, random_state=RANDOM_SEED)

    # If random search
    if args.randomsearch:

        hyperparams = random_search(train_data, val_data, test_data, train_labels, val_labels, test_labels)

        # Train again on best hyperparameters
        best_val, best_f1, threshold = train_one_model(train_data, val_data, test_data, train_labels, val_labels, test_labels, hyperparams, save=args.save)
        print('Best model val loss:', best_val)
        print('Best model f1 score:', best_f1)
        print('Best model threshold:', threshold)

    # Else train on hyperparameters, which are defined in config
    else:
        best_val, best_f1, threshold = train_one_model(train_data, val_data, test_data, train_labels, val_labels, test_labels, HYPERPARAMS, save=args.save)
