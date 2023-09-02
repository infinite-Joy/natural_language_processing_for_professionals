"""
FastAPI code
"""

from fastapi import FastAPI
import torch
from transformers import BertTokenizer, BertForSequenceClassification, TextClassificationPipeline

number_labels = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "transformer_model/"

def text_preprocessing(text):
    """
    Preprocess the text for better understanding

    """
    if isinstance(text, str):
        text = text.strip()
        text = text.lower()
    return text

def load_model():
    tokenizer = BertTokenizer.from_pretrained('transformer_model/', do_lower_case=True)
    model = BertForSequenceClassification.from_pretrained('transformer_model/', num_labels=number_labels)
    if device.type != 'cpu':
        _ = model.cuda()
    nlp_pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer)
    return nlp_pipe

nlp_pipe = load_model()
sample_text = 'I hate this game'
sample_text = text_preprocessing(sample_text)
x_dict = nlp_pipe(sample_text)
print('output of sample text', x_dict)

app = FastAPI(
    title="Sentiment Model API",
    description="A simple API that use NLP model to predict the label of game reviews",
    version="0.1",
)

@app.get("/predict-review")
def predict_sentiment(review: str):
    # perform prediction
    review = text_preprocessing(review)
    prediction = nlp_pipe(review)
    if prediction:
        return prediction