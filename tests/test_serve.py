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


def test_fastapi_server(mock_request):
    response = client.post("/predict_genre/", json=mock_request)
    assert response.status_code == 200


def test_client_server(mock_request):
    m = ModelServer(ENCODER_PATH, TRAINED_MODEL_PATH)
    # Predict a batched example
    assert m.predict(mock_request['overview'])
