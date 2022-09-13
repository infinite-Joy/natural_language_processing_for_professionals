# Base container image, which has the Ubuntu Linux distribution and Nodejs pre-installed
# command: tar -czvf jupyter.tar.gz Dockerfile config.py src/ pyproject.toml poetry.lock
FROM    ubuntu:20.04

# Install the following packages
RUN     apt-get update && apt-get install software-properties-common -y &&\
        apt-get install -y wget && apt-get install -y git &&\
        apt-get install -y zip && apt-get install -y unzip &&\
        add-apt-repository ppa:deadsnakes/ppa && apt-get update &&\
        apt-get install python3.8 -y &&apt install python3-pip -y &&\
        pip3 install --upgrade pip && pip3 install jupyter &&\
        mkdir /usr/local/notebooks

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1
# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

WORKDIR /app

# models are already downloaded
#RUN wget http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Video_Games_5.json.gz
#RUN wget https://nlp.stanford.edu/data/glove.6B.zip && unzip glove.6B.zip
# RUN wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip && unzip wiki.en.zip
# RUN git clone https://github.com/infinite-Joy/natural_language_processing_for_professionals.git

ADD . /app

RUN gunzip Video_Games_5.json.gz
RUN unzip glove.6B.zip
#RUN unzip wiki.en.zip

RUN pip install poetry==1.1.12
# RUN pip install poetry==1.1.7
RUN poetry config virtualenvs.create false \
  && poetry install --no-interaction --no-ansi

# Download required data
RUN python3 -m spacy download en_core_web_sm
RUN python3 -m nltk.downloader averaged_perceptron_tagger
RUN python3 -m nltk.downloader stopwords
RUN python3 -m nltk.downloader punkt
RUN python3 -m nltk.downloader wordnet

# Add configuration file
COPY     config.py /root/.jupyter/jupyter_notebook_config.py

# Add ipynb files
# this will work because src is part of the tar file
COPY     src/ /usr/local/notebooks/

# download models
RUN python3 load_use.py

# data preprocessing
RUN python3 process_to_csv.py
