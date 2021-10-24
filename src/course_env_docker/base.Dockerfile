FROM python:3.8

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1
# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

WORKDIR /app

RUN wget http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Video_Games_5.json.gz \
	&& gunzip Video_Games_5.json.gz
RUN wget https://nlp.stanford.edu/data/glove.6B.zip && unzip glove.6B.zip
RUN git clone https://github.com/infinite-Joy/natural_language_processing_for_professionals.git

ADD . /app

RUN ls -ltr

RUN pip install poetry==1.1.7
RUN poetry config virtualenvs.create false \
  && poetry install --no-interaction --no-ansi

# Download required data
RUN python -m spacy download en_core_web_sm
RUN python -m nltk.downloader averaged_perceptron_tagger
RUN python -m nltk.downloader stopwords
RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader wordnet
RUN python -m nltk.downloader wordnet