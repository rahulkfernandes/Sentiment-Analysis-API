import re
from pathlib import Path
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE_DIR = Path(__file__).resolve(strict=True).parent
MODEL = f'{BASE_DIR}/roberta_senti_tuned_2022-12-23'

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model =  AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)
nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

label = {'LABEL_0': 'Negative', 'LABEL_1': 'Positive'}

def preprocessing(text):
        text = re.sub(r'[\n]', '', text)
        text = re.sub(r'@[A-Za-z0-9]+', '', text)
        text = re.sub(r'#', '', text)
        text = re.sub(r'RT[\s]', '', text)
        text = re.sub(r'https?:\/\/\S+','', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'[ ]+',' ', text)
        return text.strip().lower()

def predict_pipeline(text):
    processed_text = preprocessing(text) 
    pred = nlp(processed_text)
    return label[pred[0]['label']]
