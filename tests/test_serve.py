import pytest

from config import ENCODER_PATH, TRAINED_MODEL_PATH
from serve import ModelServer


from starlette.testclient import TestClient

from rest_service.server_rest import app

client = TestClient(app)


@pytest.fixture
def mock_request():
    return {
        'title': ['New movie'],
        'overview': ['Chris was a software developer, he found a strange door in his basement and something came out of it'],
    }


@pytest.mark.parametrize("expected", [
    ({
        'title': 'New movie',
        'overview': 'Chris was a software developer, he found a strange door in his basement and something came out of it',
        'genre': 'Thriller, Horror'
    }),
])
def test_fastapi_server(mock_request, expected):
    response = client.post("/predict_genre/", json=mock_request)
    assert response.status_code == 200
    # [0] because it's always batched
    assert response.json()[0] == expected


@pytest.mark.parametrize("expected", [
    (['Thriller', 'Horror']),
])
def test_client_server(mock_request, expected):
    m = ModelServer(ENCODER_PATH, TRAINED_MODEL_PATH)
    # Predict a batched example
    assert m.predict(mock_request['overview'])[0] == expected
