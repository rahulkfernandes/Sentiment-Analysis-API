from fastapi import FastAPI
from pydantic import BaseModel
from model_helper import predict_pipeline

app = FastAPI()

class TextIn(BaseModel):
    text: str

@app.get('/')
def home():
    return {'health': 'OK'}

@app.post('/predict')
def predict(payload: TextIn):
    sentiment = predict_pipeline(payload.text)
    return {'Sentiment': sentiment}