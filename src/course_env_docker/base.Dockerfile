FROM python:3.8

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1
# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

WORKDIR /app

# models are already downloaded
#RUN wget http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Video_Games_5.json.gz
#RUN wget https://nlp.stanford.edu/data/glove.6B.zip && unzip glove.6B.zip

ADD . /app

RUN gunzip Video_Games_5.json.gz
RUN unzip glove.6B.zip
RUN pip install poetry==1.1.12
# RUN pip install poetry==1.1.7
RUN poetry config virtualenvs.create false \
  && poetry install --no-interaction --no-ansi

# Download required data
RUN python -m spacy download en_core_web_sm
RUN python -m nltk.downloader averaged_perceptron_tagger
RUN python -m nltk.downloader stopwords
RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader wordnet

# download models
RUN python load_use.py

# data preprocessing
RUN python process_to_csv.py