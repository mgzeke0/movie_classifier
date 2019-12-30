# -*- coding: utf-8 -*-
import json
import traceback

import requests

from config import PORT


def test():
    """
    Utility function to debug FastAPI service
    :return:
    """
    url = f'http://127.0.0.1:{PORT}/predict_genre/'
    while True:
        title = input('Insert movie title: ')
        overview = input('Insert movie description: ')
        if title in {'q', 'quit'} or overview in {'q', 'quit'}:
            break
        try:
            input_obj = {
                'title': [title],
                'overview': [overview],
            }
            res = requests.post(url, json=input_obj)
            result = json.loads(res.content)
            print(result)
        except Exception:
            traceback.print_exc()


if __name__ == '__main__':
    test()
