#!/usr/bin/env python
# coding: utf-8

import numpy as np

import pandas as pd

import tensorflow as tf
from tqdm import tqdm

from config import FEATURES_PATH, GENRES, DATASET_PATH, RAW_DATASET_PATH
from train.utils import get_encoder

tf.get_logger().setLevel('ERROR')


def convert_dataset(path, out_data_path, genres_dict):
    data = pd.read_csv(path)
    # Convert the genres (a json) into a list of dicts.
    # Example: [{'id': 28, 'name': 'Action'}, {'id': 80, 'name': 'Crime'}]
    data.genres = data.genres.apply(eval)

    # Since the genres field has various entries, some of which are clearly not genres but production houses,
    # I will define a specific taxonomy for this tasks with genres chosen by me. Also I will discard any movie that
    # Doesn't include at least one of the genres defined by me.
    # Now I will create a new column containing a list of genres ids.
    # This column will be then converted into one-hot encoding during training

    def convert_genres(col):
        """
        Given a list of dictionaries, return the id mapped to that genre.
        """
        if len(col):
            temp_l = []
            for gen in col:
                g = genres_dict.get(gen['id'])
                if g is not None:
                    temp_l.append(gen['id'])
            if len(temp_l):
                return temp_l
        return None

    data['genres_list'] = data.genres.apply(convert_genres)
    # Keep only the columns I need
    data = data.loc[:, ['id', 'title', 'overview', 'genres', 'genres_list']]
    # Drop movies without supported genres
    data.dropna(inplace=True, subset=['overview', 'genres_list'])
    data.to_csv(out_data_path, index=None)
    assert len(data[pd.isna(data['genres_list'])]) == 0


def compute_features(data_path, features_path, save_to_disk=True):
    """
    This function computes features from a Pretrained model.
    These features are then used for transfer learning
    In a downstream classification task
    """
    data = pd.read_csv(data_path)
    # Instead of installing TensorFlow hub, download a Universal Sentence Encoder model
    encoder = get_encoder()

    # Pre compute vector features
    encode_batch_size = 128
    result = []
    for i in tqdm(range(0, len(data), encode_batch_size), total=int(len(data)/encode_batch_size)):
        batch_features = encoder(tf.constant(data.overview[i:i + encode_batch_size].tolist()))
        ids = data.id[i:i + encode_batch_size].tolist()
        if save_to_disk:
            for id, f in zip(ids, batch_features):
                np.save(features_path + str(id), f.numpy())
        result.extend(batch_features)
    print('Features created')
    return result


if __name__ == '__main__':
    convert_dataset(RAW_DATASET_PATH, DATASET_PATH, GENRES)
    compute_features(DATASET_PATH, FEATURES_PATH)
