import os


# Define available genres for training

GENRES = {
    16: 'Animation',
    35: 'Comedy',
    10751: 'Family',
    12: 'Adventure',
    14: 'Fantasy',
    10749: 'Romance',
    18: 'Drama',
    28: 'Action',
    80: 'Crime',
    53: 'Thriller',
    27: 'Horror',
    36: 'History',
    878: 'Science Fiction',
    9648: 'Mystery',
    10752: 'War',
    10769: 'Foreign',
    10402: 'Music',
    99: 'Documentary',
    37: 'Western',
    10770: 'TV Movie',
}

HYPERPARAMS = {
    'num_units': 310,
    'lr': 0.004,
    'epochs': 200,
    'batch_size': 768,
    'threshold': 0.3
}

ENCODER_PATH = os.path.join('model','encoder')
RAW_DATASET_PATH = os.path.join('data', 'the-movies-dataset', 'movies_metadata.csv')
DATASET_PATH = os.path.join('data', 'training')
FEATURES_PATH = os.path.join('data', 'training', 'vectors')
TRAINED_MODEL_PATH = os.path.join('model', 'trained_model')
RANDOM_SEED = 42
PORT = 8081
