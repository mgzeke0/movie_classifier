# Movies Classifier

## Introduction

Given a description and a title, this project classifies a movie into a genre. The available genres are
'Animation', 'Comedy', 'Family', 'Adventure', 'Fantasy', 'Romance', 'Drama', 'Action', 'Crime', 'Thriller', 'Horror', 'History', 'Science Fiction', 'Mystery', 'War', 'Foreign', 'Music', 'Documentary', 'Western', 'TV Movie'.

The task is formulated as a Multi Label Classification, so a single movie description can belong to multiple independent genres.

The Machine Learning model used is composed of 2 parts: a sentence encoder and a 1 Hidden Layer Neural Network. The sentence encoder is a Universal Sentence Encoder [Universal Sentence Encoder large](https://tfhub.dev/google/universal-sentence-encoder-large/5). It's a Transformer pre-trained on various NLP tasks, with the goal of using transfer learning in downstream tasks. It is also faster and lighter than other language models such as BERT and it doesn't need explicit text preprocessing. I chose it because it should return a single fixed vector for a text of any length, where semantically similar texts will have similar vectors. These pre trained representations should be helpful in classifying similar texts in meaning. The second part is simply a 1 Hidden Layer Neural Network. After performing some tests it looks like a single layer is enough to get good results without overfitting. I explored only Fully-Connected layers and not different architectures. The two models are implemented and used separately in the code, to explicitly show the different steps (Encoder/Classifier) but it would be straightforward to save them as a single model. I train only the second step, the 1 hidden layer Neural Network, pre-computing a vector for each movie description and keeping the Encoder's weights fixed. Finally, I perform a quick random search over the space of some plausible hyperparameters.

The inference part can be run in a single command line or as a REST server.

To be able to run it, please read the whole file and follow all the Set up and Preparation sections.

## Set up

### Python environment

- This project was developed using Python 3.6 with Anaconda (Miniconda). Install it if you don't have it https://docs.conda.io/projects/conda/en/latest/user-guide/install/
    - Alternatively, install Python 3.6 from https://www.python.org/downloads/ but you'd have to install other packages like pip.
- (Optional) I recommend using a virtual environment, I use Anaconda, but if you didn't install it you can use Virtualenv.
    - With Anaconda, run `conda create -n movie_classifier python=3.6` and `conda activate movie_classifier`

Once the Python environment is set up, simply run To install the project's requirements

`pip install -r requirements.txt`


## Preparation

Some files that are external to git are needed. In order to be able to run tests, train a model and serve a trained model, perform the following steps.

I suggest to prepare the folders by running

`mkdir -p model/encoder model/trained_model data/the-movies-dataset data/training`

#### Data
(Only needed for training)

Download the training data: [The Movies Dataset](https://www.kaggle.com/rounakbanik/the-movies-dataset/data)

extract it and place it under the folder `data/the-movies-dataset/`.

#### Sentence Embedding model

Download [Universal Sentence Encoder large](https://tfhub.dev/google/universal-sentence-encoder-large/5?tf-hub-format=compressed)

Extract it and place it under the folder `model/encoder/`

#### Trained model

Download a trained model from [Google Drive](https://drive.google.com/open?id=13vMbEIbPnvGjawAuoBRTA3gtBC629Nt0)

Again, unpack the files under `model/trained_model/`.

After extracting data, encoder and the trained model, your new folders should look like this:

- model
  - encoder
    - variables
    - saved_model.pb
    - assets
  - trained_model
    - saved_model.pb
    - assets
    - variables
    - reverse_dict.pkl
    - hyperparams.pkl

- data
  - the-movies-dataset
    - credits.csv
    - keywords.csv
    - links.csv
    - links_small.csv
    - movies_metadata.csv
    - ratings.csv
    - ratings_small.csv

Now you can run tests

`pytest`

## Training
### Pre compute features

Prepare data and extract vectors for the single reviews by running

`python -m train.prepare_dataset --vectors`

This script first creates labels that are in a suitable format for ML models, then pre-computes and saves on disk a vector for each movie overview. The vectors are computed from the Universal Sentence Encoder downloaded before.

You can find the pre-computed vectors [here](https://drive.google.com/open?id=1OiCfqyvJ43VSfJdkmVTl5qNV7tJcmVRc) to save time. 
The folder `vectors` should be placed under `data/training/`
Then just run `python -m train.prepare_dataset --vectors`

### Run Training

After creating the pre computed vectors, train an additional model that classifies them. Run

`python -m train.run_training`

To replicate my results. You should get as output 

```
Val data f1 score: 0.55821985
Val data f1 score tuned: 0.6113644
Test data f1 score: 0.5571354
Test data f1 score tuned: 0.6085857
```
Even though the values may vary a little.

To perform a small random search for parameters, run

`python -m train.run_training --randomsearch`

The script will also save the model under `model/trained_model/`. 

Pass the flag --save `python -m train.run_training --save` to save it.


## Inference

There are 3 ways to do inference:
1) Run the command line application
2) Run a REST server and call it with a Python client
3) Build and run a Docker image

### Python command line

Run the command specifying a title and an overview.

`python -m movie_classifier --title TITLE --overview OVERVIEW`

### REST Service

Run 

`uvicorn rest_service.server_rest:app --host "0.0.0.0" --port 8081`

And in a separate window

`python -m rest_service.client_rest`


### Docker image

Install Docker https://docs.docker.com/install/

Build the Docker image running

`docker build -t movie_classifier .`

Run the container

`docker run -p 8081:8081 movie_classifier:latest`

Run the client

`python -m rest_service.client_rest`
