import os

import tensorflow as tf

from config import ENCODER_PATH


def get_encoder():
    try:
        loaded = tf.saved_model.load(ENCODER_PATH)
        print('Loaded Universal Sentence Encoder')
        return loaded
    except OSError:
        # If the file is not found, download it from my Gdrive
        raise Exception(
            f'Universal Sentence Encoder not found, download it here: https://drive.google.com/open?id=13u2KK8Fq94Q8Qv0xEGhi69zAH-qwCsL5'
            f'And place it in {ENCODER_PATH.split(os.path.sep)[0]}'
        )
