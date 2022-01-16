import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification, TextClassificationPipeline


# the model we gonna train, base uncased BERT
model_name = "transformer_model/"
max_length = 128

st.cache(show_spinner=False)
def load_model():
    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=5)
    nlp_pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer)
    return nlp_pipe

npl_pipe = load_model()

# specify the UI elements
st.header("Prototyping an NLP solution")
st.text("This demo uses a model for classsification.")
add_text_sidebar = st.sidebar.title("Menu")
add_text_sidebar = st.sidebar.text("Just some random text.")

text = st.text_input(label='Insert the text')

def text_preprocessing(text):
    """
    Preprocess the text for better understanding

    """
    if isinstance(text, str):
        text = text.strip()
        text = text.lower()
    return text

if len(text) > 0:
    text = text_preprocessing(text)
    x_dict = npl_pipe(text)
    st.text('Answer: ' + x_dict[0]['label'])
