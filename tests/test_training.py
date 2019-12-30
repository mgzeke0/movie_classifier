import numpy as np

import pytest

import tensorflow as tf

from train.train_functions import f1_score, train_one_model, one_hot

from config import GENRES


@pytest.fixture
def random_data():
    """
    Return random training data to feed the model.
    Train, valid and test data and labels
    :return:
    """
    np.random.seed(42)
    return (
        np.random.randn(10, 512),
        np.random.randn(10, 512),
        np.random.randn(10, 512),
        np.random.randint(0, 2, size=[10, len(GENRES)]),
        np.random.randint(0, 2, size=[10, len(GENRES)]),
        np.random.randint(0, 2, size=[10, len(GENRES)]),
    )


@pytest.fixture
def hyperparams():
    """
    Return an instance of hyperparameters to build  a model
    :return:
    """
    return {'num_units': 710, 'lr': 0.09707718827433366, 'epochs': 200, 'batch_size': 2, 'early_stop_delta': 0.003743537084091529, 'patience': 15}


def test_train_one_model(random_data, hyperparams):
    best_val_loss, best_f1, best_thresh = train_one_model(*random_data, hyperparams=hyperparams, random_seed=42, verbose=0)

    assert np.allclose(12.06, best_val_loss, atol=0.01)
    assert np.allclose(0.62, best_f1, atol=0.01)
    assert np.allclose(0.1, best_thresh, atol=0.01)


@pytest.mark.parametrize("y,yhat,expected", [
        (tf.constant([[1, 1, 0]], dtype=tf.float32), tf.constant([[0.1, 0.2, 0.6]]), 0),
        (tf.constant([[0, 0, 1]], dtype=tf.float32), tf.constant([[0.1, 0.2, 0.6]]), 1),
        (tf.constant([[0, 1, 1]], dtype=tf.float32), tf.constant([[0.1, 0.2, 0.6]]), 0.6666666),
])
def test_f1_score(y, yhat, expected):
    assert np.allclose(f1_score(y, yhat, thresh=0.5).numpy(), expected)


@pytest.mark.parametrize("labels, num_classes, labels_list, expected", [
    ([['a', 'b'], ['c']], 3, ['a', 'b', 'c'], [[1, 1, 0], [0, 0, 1]])
])
def test_one_hot(labels, num_classes, labels_list, expected):
    assert np.allclose(one_hot(labels, num_classes, labels_list)[0], expected)
