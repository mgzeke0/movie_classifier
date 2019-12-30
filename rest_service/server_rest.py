from typing import Dict, List

import serve

from fastapi import FastAPI

from pydantic import BaseModel

from config import ENCODER_PATH, TRAINED_MODEL_PATH


app = FastAPI()

model = serve.ModelServer(ENCODER_PATH, TRAINED_MODEL_PATH)


@app.post('/predict_genre/', response_model=List[Dict[str, str]])
async def predict_genre(title: List[str], overview: List[str]) -> List[Dict[str, str]]:
    result = model.predict(overview)
    output = []
    for i, res in enumerate(result):
        output.append({
            'title': title[i],
            'overview': overview[i],
            'genre': ', '.join(res),
        })
    return output
