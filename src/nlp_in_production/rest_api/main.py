# text preprocessing modules
# text preprocessing modules
from fastapi import FastAPI 
import torch
from transformers import BertTokenizer, BertForSequenceClassification, TextClassificationPipeline

number_labels = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 128
batch_size = 32

app = FastAPI(
    title="Sentiment Model API",
    description="A simple API that use NLP model to predict the label of game reviews",
    version="0.1",
)



def load_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    model = BertForSequenceClassification.from_pretrained('models/transformer_model/', num_labels=number_labels)
    if device.type != 'cpu':
        _ = model.cuda()
    nlp_pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer)
    return nlp_pipe

nlp_pipe = load_model()

text = '''greeting fellow amazonians i bought this game thinking that on my down time from carnival life i would get some stress relief from all the rubes that try to win prizes from me. boy was i wrong! first, mongo plays this game all the time and he is always late for his shift at the fun house, second when he does go to work he doesn't collect any tickets he just points his fingers at little kids and says "bang! bang!", but what took the cake was one day as he was wheeling his rascal to the fun house he grabbed a squirt gun from the "fill the clowns mouth with water game" and started shooting the winner of the ms. concordia pageant and her prized pig. he thought that since he had unlimited ammo in max payne that the water would never stop unfortunately for him our fellow carney c-dog turned off the hose before mongo knew what happened. luckely i was able to smooth everything over with ms. concordia (i kissed her pig in return for not pressing charges). i may have rated this game better but i warn people watching over autistic kids do not let them play this game'''

x_dict = nlp_pipe(text)
print(x_dict)

x_dict = nlp_pipe('I hate this game')
print(x_dict)

@app.get("/predict-review")
def predict_sentiment(review: str):
    """
    A simple function that receive a review content and predict the label of the text.

    :param review:
    :return: prediction, probabilities
    """
    # perform prediction
    prediction = nlp_pipe(review)
    if prediction:
        return prediction