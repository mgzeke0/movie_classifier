# Movies Classifier

## Intro
Given a description and a title, this project classifies a movie into a genre. The available genres are
'Animation', 'Comedy', 'Family', 'Adventure', 'Fantasy', 'Romance', 'Drama', 'Action', 'Crime', 'Thriller', 'Horror', 'History', 'Science Fiction', 'Mystery', 'War', 'Foreign', 'Music', 'Documentary', 'Western', 'TV Movie'

## Set up
### Python environment
- This project was developed using Python 3.6 with Anaconda (Miniconda). Install it if you don't have it https://docs.conda.io/projects/conda/en/latest/user-guide/install/
    - Alternatively, install Python 3.6 from https://www.python.org/downloads/ but you'd have to install other packages like pip.
- (Optional) I recommend using a virtual environment, I use Anaconda, but if you didn't install it you can use Virtualenv.
    - With Anaconda, run `conda create -n movie_classifier python=3.6` and `conda activate movie_classifier`

Once the Python environment is set up, simply run To install the project's requirements

`pip install -r requirements.txt`


## Preparation
#### Data
Download the training data: [The Movies Dataset](https://www.kaggle.com/rounakbanik/the-movies-dataset/data)

extract it and place it under the folder `data/the-movies-dataset/`.

#### Embedding model
Download https://tfhub.dev/google/universal-sentence-encoder-large/5?tf-hub-format=compressed

Extract it and place it under the folder `model/encoder/`

#### Trained model
Download a trained model from [Google Drive](https://drive.google.com/open?id=13vMbEIbPnvGjawAuoBRTA3gtBC629Nt0)

Again, unpack the files under `model/trained_model`. The folder structure should look like


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



### Pre compute features
Prepare data and extract vectors for the single reviews by running 

`python-m train.prepare_dataset`

This script first creates labels that are in a suitable format for ML models, then pre-computes and saves on disk a vector for each movie overview. The vectors are computed from the Universal Sentence Encoder downloaded before.

### Run Training
After creating the pre computed vectors, train an additional model that classifies them. Run

`python -m train.run_training`

To perform a small random search for parameters and replicate my results. The script will also save the model under `model/trained_model/`

Otherwise use the function `train_one_model` for a single training.

## Inference
There are 3 ways to do inference:
1) Build and run a Docker image (also requires Python3.6)
2) Run a REST server and call it with a Python client
3) Run the command line application

### Model
If you didn't train a model, download a trained model from [Google Drive](https://drive.google.com/open?id=13vMbEIbPnvGjawAuoBRTA3gtBC629Nt0)

Again, unpack the files under `model/trained_model`. The folder structure should look like
- model
  - trained_model
    - saved_model.pb
    - assets
    - variables
    - reverse_dict.pkl
    - hyperparams.pkl

### Docker image
Install Docker https://docs.docker.com/install/

Build the Docker image running

`docker build -t movie_classifier .`

Run the container

`docker run -p 8081:8081 movie_classifier:latest`

Run the client

`python -m rest_service.client_rest`

### REST Service

Run 

`python -m rest_service.server_rest`

And in a separate window

`python -m rest_service.client_rest`


### Python command line

Run the command specifying a title and an overview.

`python -m movie_classifier --title TITLE --overview OVERVIEW`
