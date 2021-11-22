import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification, TextClassificationPipeline


# the model we gonna train, base uncased BERT
# check text classification models here: https://huggingface.co/models?filter=text-classification
model_name = "bert-base-uncased"
model_name = "rest_api/models/transformer_model/"
# max sequence length for each document/sentence sample
max_length = 128


st.cache(show_spinner=False)
def load_model():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=5)
    nlp_pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer)
    return nlp_pipe

npl_pipe = load_model()

st.header("Prototyping an NLP solution")
st.text("This demo uses a model for classsification.")
add_text_sidebar = st.sidebar.title("Menu")
add_text_sidebar = st.sidebar.text("Just some random text.")

text = st.text_input(label='Insert the text')

if len(text) > 0:
    x_dict = npl_pipe(text)
    st.text('Answer: ' + x_dict[0]['label'])
