import json
import os
import pickle

import numpy as np

import tensorflow as tf

from config import FEATURES_PATH, GENRES, RANDOM_SEED, TRAINED_MODEL_PATH
from train.models_class import MultiLabelClassifier


def load_batches(ids):
    return np.array([np.load(os.path.join(FEATURES_PATH, str(i) + '.npy')) for i in ids])


def one_hot(labels, num_classes, labels_list):
    """
    Create a one hot representation of classes, in a multi-label fashion
    [[1, 3, 4], [2, 3]] --> [[1, 0, 1, 1], [0, 1, 1, 0]]
    :param labels: list of lists
    :param num_classes: total number of classes
    :param labels_list: the possible labels
    :return: a one hot encoded representation and a dict to map indexes to classes
    """
    output = np.zeros((len(labels), num_classes), dtype=np.int32)
    one_hot_dict = {k: i for i, k in enumerate(labels_list)}
    for i, l in enumerate(labels):
        for l_ in l:
            output[i][one_hot_dict[l_]] = 1
    return output, {v: k for k, v in one_hot_dict.items()}


def f1_score(y, y_hat, thresh=0.5):
    # Derived from https://gist.github.com/ashrefm/fc7925e7abbd695bb5b0159cf154a8f7#file-macro_f1-py
    """Compute the micro F1-score on a batch of observations (average F1 across labels)

    Args:
        y (int32 Tensor): labels array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
        thresh: probability value above which we predict positive

    Returns:
        micro_f1 (scalar Tensor): value of micro F1 for the batch
    """
    y_pred = tf.cast(tf.greater(y_hat, thresh), tf.float32)
    tp = tf.cast(tf.math.count_nonzero(y_pred * y), tf.float32)
    fp = tf.cast(tf.math.count_nonzero(y_pred * (1 - y)), tf.float32)
    fn = tf.cast(tf.math.count_nonzero((1 - y_pred) * y), tf.float32)
    f1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
    macro_f1 = tf.reduce_mean(f1)
    return macro_f1


def train_one_model(train_data, val_data, test_data, train_labels, val_labels, test_labels, hyperparams, save=False, random_seed=RANDOM_SEED, verbose=1):
    """
    Train one model.
    the hyperparams argument should be a dict that requires
        'num_units': number of nodes in the hidden layer
        'lr': learning rate for Adam
        'epochs': number of max epochs
        'batch_size': number of mini batch examples

    :param train_data: Training features (batch_size, dim)
    :param val_data: Validation features (batch_size, dim)
    :param test_data: Test features (batch_size, dim)
    :param train_labels: Training labels (one-hot) (batch_size, num_classes)
    :param val_labels: Validation labels (one-hot) (batch_size, num_classes)
    :param test_labels: Test labels (one-hot) (batch_size, num_classes)
    :param hyperparams: dict of hyperparameters
    :param save: Whether to save the model
    :param random_seed: Seed for reproducibility
    :param verbose: Keras verbose parameter. If 0 suppress every print in this function
    :return: (best_val_loss, best_f1, best_thresh) best validation loss, best f1 score and tuned threshold
    """

    tf.random.set_seed(random_seed)

    # Define the model hyperparameters
    model = MultiLabelClassifier(len(GENRES), hyperparams['num_units'], tf.nn.sigmoid)

    # Get the trainable part
    trainable_model = model.trainable_model

    # Define an optimizer
    optimizer = tf.optimizers.Adam(learning_rate=hyperparams['lr'])

    # Compile the model
    trainable_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', f1_score])

    # Define an early stopping criterion
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True,
                                                      min_delta=0.0001)

    epochs = hyperparams['epochs']

    # Perform model training
    history = trainable_model.fit(
        train_data,
        train_labels,
        batch_size=hyperparams['batch_size'],
        epochs=epochs,
        verbose=verbose,
        validation_data=(val_data, val_labels),
        callbacks=[early_stopping],
    )
    # Get validation predictions and find a threshold that maximise validation f1 score
    val_predictions = trainable_model.predict(val_data)
    best_score = 0
    best_thresh = 0.5
    for i in range(1, 9):
        t = i*0.1
        s = f1_score(val_labels, val_predictions, thresh=t).numpy()
        if s > best_score:
            best_score = s
            best_thresh = t

    # Compute f1 scores with tuned threshold
    tuned_val_f1_score = f1_score(val_labels, val_predictions, thresh=best_thresh).numpy()
    tuned_test_f1_score = f1_score(test_labels, trainable_model.predict(test_data), thresh=best_thresh).numpy()
    if verbose != 0:
        print('Val data f1 score:', f1_score(val_labels, val_predictions).numpy())
        print('Val data f1 score tuned:', tuned_val_f1_score)
        print('Test data f1 score:', f1_score(test_labels, trainable_model.predict(test_data)).numpy())
        print('Test data f1 score tuned:', tuned_test_f1_score)

    # Prepare return values
    e = early_stopping.stopped_epoch
    best_val_loss = history.history['val_loss'][e]

    if save:
        # Save the tensorflow model in the TF 2.0 format
        trainable_model.save(TRAINED_MODEL_PATH, include_optimizer=False, save_format='tf')

        # Save hyperparameters dictionary
        hyperparams['threshold'] = best_thresh
        pickle.dump(hyperparams, open(os.path.join(TRAINED_MODEL_PATH , 'hyperparams.pkl'), 'wb'))

    # Return the validation loss at the best epoch, the tuned validation f1 score and the optimized threshold
    return best_val_loss, tuned_val_f1_score, best_thresh


def random_search(train_data, val_data, test_data, train_labels, val_labels, test_labels):
    """
    To simplify hyperparameter search, this function allows a small random search over a subset of
    plausible hand-coded hyperparameter choices
    :param train_data: Training features (batch_size, dim)
    :param val_data: Validation features (batch_size, dim)
    :param test_data: Test features (batch_size, dim)
    :param train_labels: Training labels (one-hot) (batch_size, num_classes)
    :param val_labels: Validation labels (one-hot) (batch_size, num_classes)
    :param test_labels: Test labels (one-hot) (batch_size, num_classes)
    :return:
    """
    random_combinations = 25

    # Pre allocate random values
    choice = {
        'num_units': np.random.choice(list(range(10, 1000, 100)), size=random_combinations),
        'lr': np.random.random_sample(size=random_combinations) * (0.1 - 0.000001) + 0.000001,
        'epochs': [200],
        'batch_size': np.random.choice([16, 32, 64, 128, 256, 512, 768, 1024], size=random_combinations),
    }
    results = {
        'val_loss': [],
        'val_f1': [],
        'hyperparams': [],
        'threshold': [],
    }
    # Run as many trainings as hyper parameters combinations
    for i in range(random_combinations):
        hyperparams = {}
        for k, v in choice.items():
            hyperparams[k] = v[i % len(v)]
        print('Training with', hyperparams)
        val_l, val_f1, thresh = train_one_model(train_data, val_data, test_data, train_labels, val_labels, test_labels, hyperparams)
        results['val_loss'].append(val_l)
        results['val_f1'].append(val_f1)
        results['threshold'].append(thresh)
        results['hyperparams'].append(hyperparams)

    min_loss_i = int(np.argmin(results['val_loss']))
    print('-----------------------')
    print('Best val loss:', results['val_loss'][min_loss_i])
    print('Best untuned f1 score:', results['val_f1'][min_loss_i])
    print(results['hyperparams'][min_loss_i])
    print(results['threshold'][min_loss_i])
    return results['hyperparams'][min_loss_i]
