import pandas as pd

import pytest

from train.prepare_dataset import convert_dataset, compute_features


@pytest.fixture
def mock_data():
    genres = [{'id': 53, 'name': 'Thriller'}, {'id': 27, 'name': 'Horror'}]
    overview = 'Chris was a software developer, he found a strange door in his basement and something came out of it'
    title = 'New movie'
    id_ = 1
    data = pd.DataFrame([{
        'id': id_,
        'title': title,
        'overview': overview,
        'genres': genres,
    }])
    genres_dict = {
        53: 'Thriller',
        27: 'Horror',
    }
    return data, genres_dict


def test_convert_dataset(mock_data, tmpdir):

    dataframe, genres_dict = mock_data

    # Prepare a temporary file to load
    datapath = tmpdir.mkdir("data")
    dataframe.to_csv(datapath + 'data.csv', index=None)
    convert_dataset(path=datapath + 'data.csv', out_data_path=datapath + 'data_preprocessed.csv', genres_dict=genres_dict)
    data = pd.read_csv(datapath + 'data_preprocessed.csv')
    assert data['genres_list'].apply(eval).tolist()[0] == [53, 27]


def test_create_features(mock_data, tmpdir):
    dataframe, genres_dict = mock_data

    # Prepare a temporary file to load
    datapath = tmpdir.mkdir("data/")
    dataframe.to_csv(datapath + 'data.csv', index=None)
    assert compute_features(str(datapath) + 'data.csv', datapath, save_to_disk=False)
