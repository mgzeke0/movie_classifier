import argparse

from config import ENCODER_PATH, TRAINED_MODEL_PATH
from serve import ModelServer


parser = argparse.ArgumentParser()
parser.add_argument('--title', type=str, help='Movie title', required=True)
parser.add_argument('--overview', help='Movie description', required=True)

args = parser.parse_args()

m = ModelServer(ENCODER_PATH, TRAINED_MODEL_PATH)

# Predict a batched example
predictions = m.predict([args.overview])
for pred in predictions:
    print({
        'title': args.title,
        'description': args.overview,
        'genre': ', '.join(pred),
    })
