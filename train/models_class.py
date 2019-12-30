import tensorflow as tf


class MultiLabelClassifier:
    def __init__(self, num_classes, num_units, activation):
        self.num_classes = num_classes
        self.activation = activation
        self.num_units = num_units
        self.trainable_model = self.get_trainable_model()

    def get_trainable_model(self):
        return tf.keras.Sequential([
            tf.keras.layers.Dense(self.num_units, activation=self.activation),
            tf.keras.layers.Dense(self.num_classes, activation=tf.nn.sigmoid),
        ])
