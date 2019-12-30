import pickle

import tensorflow as tf

from config import GENRES

tf.get_logger().setLevel('ERROR')


class ModelServer:
    def __init__(self, ENCODER_PATH, TRAINED_MODEL_PATH):
        self.encoder = tf.saved_model.load(ENCODER_PATH)
        self.trained_model = tf.saved_model.load(TRAINED_MODEL_PATH)
        self.reverse_dict = pickle.load(open(TRAINED_MODEL_PATH + 'reverse_dict.pkl', 'rb'))
        self.threshold = pickle.load(open(TRAINED_MODEL_PATH + 'hyperparams.pkl', 'rb'))['threshold']

    def predict(self, description):
        f = self.encoder(tf.constant(description))
        predictions = self.trained_model(f)
        preds = tf.cast(tf.greater(predictions, self.threshold), tf.float32)
        genres = []
        for single_sentence in preds:
            temp = []
            for i, p in enumerate(single_sentence):
                if p == 1:
                    temp.append(GENRES[self.reverse_dict[i]])
            genres.append(temp)
        return genres
