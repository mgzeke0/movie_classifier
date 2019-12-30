from train.utils import get_encoder


def test_get_encoder():
    """
    Just test that the functions returns something and doesn't crash
    """
    assert get_encoder()
