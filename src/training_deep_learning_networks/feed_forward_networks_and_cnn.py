# -*- coding: utf-8 -*-
"""feed_forward_networks_and_cnn

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1xZYjNazv1mow7pXY5JfEWXT9AaS4Ol8s
"""

from google.colab import drive
drive.mount('/content/drive')

!ls -ltr /content/drive/MyDrive/educative_natural_language_processing_for_professionals/models

"""## Gather Data"""

import urllib.request as req
from urllib.parse import urlparse
import os
import progressbar
import zipfile
import gzip
import shutil
import json
import pandas as pd
import re
import string
import imblearn

pbar = None


def show_progress(block_num, block_size, total_size):
    global pbar
    if pbar is None:
        pbar = progressbar.ProgressBar(maxval=total_size)
        pbar.start()

    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(downloaded)
    else:
        pbar.finish()
        pbar = None

def wget(url):
    a = urlparse(url)
    filename = os.path.basename(a.path)
    if not os.path.isfile(filename):
        req.urlretrieve(url, filename, show_progress)
        print(f'downloaded to {filename}')
    else:
        print(f'file {filename} has already been downloaded')
    return filename

def unzip(filename, directory_to_extract_to=os.getcwd()):
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)
        print(f'extraction done {zip_ref.namelist()}')

def gunzip(gzfile, fout):
    with gzip.open(gzfile, 'rb') as f_in:
        with open(fout, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f'{gzfile} extracted to {fout}')


# map punctuation to space
translator = str.maketrans(string.punctuation, ' '*len(string.punctuation)) 

def text_preprocessing(text):
    """
    Preprocess the text for better understanding
    
    """
    text = text.strip()
    text = text.lower()
    text = text.translate(translator)
    return text


filename = wget("https://nlp.stanford.edu/data/glove.6B.zip")
unzip(filename)
Video_Games_5 = wget('http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Video_Games_5.json.gz')
df = pd.read_json("./Video_Games_5.json.gz", lines=True, compression='gzip')
df = df[['reviewText', 'overall']]
df = df.dropna()
df['reviewText'] = df['reviewText'].apply(text_preprocessing)
df = df.drop_duplicates()
print(df.shape)

df.sample(10)

"""## Split train and test"""

from sklearn.model_selection import train_test_split

sentences = df['reviewText'].values
y = df['overall'].values
y = pd.get_dummies(y).values

sentences_train, sentences_test, y_train, y_test = train_test_split(
    sentences, y, test_size=0.3, random_state=42, stratify=df.overall)

print(len(sentences_train), len(sentences_test))

"""## Input Tensors"""

from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(sentences_train)

X_train = tokenizer.texts_to_sequences(sentences_train)
X_test = tokenizer.texts_to_sequences(sentences_test)

vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

from tensorflow.keras.preprocessing.sequence import pad_sequences

maxlen = 100

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

print(X_train[0, :])

"""## Embedding Matrix

Mapping of words in the corpus with the glove embeddings
"""

import numpy as np

def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath) as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word] 
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix

embedding_dim = 50
embedding_matrix = create_embedding_matrix(
    './glove.6B.50d.txt', tokenizer.word_index, embedding_dim)

"""## Compute class weights"""

from sklearn.utils import compute_class_weight

y_integers = np.argmax(y_train, axis=1)
class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
d_class_weights = dict(enumerate(class_weights))

"""## helper functions"""

import matplotlib.pyplot as plt
plt.style.use('ggplot')

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

"""## Shallow Neural Network"""

import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow.keras.optimizers import Adam

maxlen = 100

model = Sequential()
model.add(layers.Embedding(vocab_size, embedding_dim, 
                           weights=[embedding_matrix], 
                           input_length=maxlen, 
                           trainable=False))
model.add(layers.Dropout(0.2))
model.add(layers.GlobalAveragePooling1D())
model.add(layers.Dropout(0.2))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(5, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

batch_size = 64
epochs = 100

history = model.fit(X_train, y_train, 
                    batch_size=batch_size, epochs=epochs, 
                    class_weight=d_class_weights, shuffle=True,
                    verbose=1, validation_split=0.1)

import matplotlib.pyplot as plt
plt.style.use('ggplot')

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

plot_history(history)

loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

import string

# # map punctuation to space
# translator = str.maketrans(string.punctuation, ' '*len(string.punctuation)) 

# def text_preprocessing(text):
#     """
#     Preprocess the text for better understanding
    
#     """
#     text = text.strip()
#     text = text.lower()
#     text = text.translate(translator)
#     return text

def predict(text, model, tokenizer):
    orig_text = text
    text = text_preprocessing(text)
    text = [text]
    X = tokenizer.texts_to_sequences(text)
    X = pad_sequences(X, padding='post', maxlen=maxlen)
    # print(np.sum(model.predict(X)[0]))
    probs = model.predict(X)[0]
    # assert np.sum(probs) == 1.0
    prediction = np.argmax(probs)
    print('input text: ', orig_text)
    print('predicted class: ', prediction + 1, 'confidence:', np.max(probs))


predict('I liked the product quite a lot.', model, tokenizer)
predict('playing this was the worst time in my life.', model, tokenizer)

from sklearn.metrics import accuracy_score, matthews_corrcoef
from imblearn.metrics import classification_report_imbalanced

y_pred = model.predict(X_test)

print('accuracy:', accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)))
print('matthews_corrcoef:', matthews_corrcoef(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)))
print('classification_report:\n', classification_report_imbalanced(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)))

"""### Plot of model metrics"""

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})

# data
x = ["CV+NB",             "FF"]
y = [0.33504969057289086, 0.2248240429789915]

ax = sns.barplot(x=x, y=y)

"""## Convolutional Neural Networks"""

import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow.keras import layers

tf.random.set_seed(42)


model = Sequential()
model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen, weights=[embedding_matrix], trainable=False))
model.add(layers.Conv1D(128, 3, activation='relu'))
# TODO try by using batch normalisation as well
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(5, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())

"""### training the base CNN model"""

batch_size = 64
epochs = 100

history = model.fit(X_train, y_train, 
                    batch_size=batch_size, epochs=epochs, 
                    class_weight=d_class_weights, shuffle=True,
                    verbose=1, validation_split=0.1)

"""### plot training statistics"""

import matplotlib.pyplot as plt
plt.style.use('ggplot')

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

plot_history(history)

loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

"""### predict function for the trigram model"""

import string

def predict(text, model, tokenizer):
    orig_text = text
    text = text_preprocessing(text)
    text = [text]
    X = tokenizer.texts_to_sequences(text)
    X = pad_sequences(X, padding='post', maxlen=maxlen)
    # print(np.sum(model.predict(X)[0]))
    probs = model.predict(X)[0]
    # assert np.sum(probs) == 1.0
    prediction = np.argmax(probs)
    print('input text: ', orig_text)
    print('predicted class: ', prediction + 1, 'confidence:', np.max(probs))

predict('I liked the product quite a lot.', model, tokenizer)
predict('playing this was the worst time in my life.', model, tokenizer)

"""### metrics for the trigram model"""

from sklearn.metrics import accuracy_score, matthews_corrcoef
from imblearn.metrics import classification_report_imbalanced

y_pred = model.predict(X_test)

print('accuracy:', accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)))
print('matthews_corrcoef:', matthews_corrcoef(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)))
print('classification_report:\n', classification_report_imbalanced(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)))

"""### plot of model metrics till now"""

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})

# data
x = ["CV+NB",             "FF",               "CNN_trigram"]
y = [0.33504969057289086, 0.2148927270393169, 0.3120980983604428]

ax = sns.barplot(x=x, y=y)

"""### save the trigram model"""

trigram_model_path = "/content/drive/MyDrive/educative_natural_language_processing_for_professionals/models/cnn_trigram_model"
model.save(trigram_model_path)

"""## CNN234GramModel

some bigrams such as "Final Fantasy" is a strong indicator that the reviews are good.
"""

df[df.reviewText.str.contains("final fantasy")].sample(10)

import string
import tensorflow as tf
from tensorflow.keras import Sequential, layers, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, matthews_corrcoef
from imblearn.metrics import classification_report_imbalanced
import matplotlib.pyplot as plt
plt.style.use('ggplot')


# map punctuation to space
translator = str.maketrans(string.punctuation, ' '*len(string.punctuation)) 

def text_preprocessing(text):
    """
    Preprocess the text for better understanding
    
    """
    text = text.strip()
    text = text.lower()
    text = text.translate(translator)
    return text

def predict(text, model, tokenizer):
    orig_text = text
    text = text_preprocessing(text)
    text = [text]
    X = tokenizer.texts_to_sequences(text)
    X = pad_sequences(X, padding='post', maxlen=maxlen)
    # print(np.sum(model.predict(X)[0]))
    probs = model.predict(X)[0]
    # assert np.sum(probs) == 1.0
    prediction = np.argmax(probs)
    print('input text: ', orig_text)
    print('predicted class: ', prediction + 1, 'confidence:', np.max(probs))

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()


class CNN234GramModel(tf.keras.Model):
    def __init__(self, nb_filters=50,
                 FFN_units=512, nb_classes=2, dropout_rate=0.1, name="dcnn"):
        super(CNN234GramModel, self).__init__(name=name)
        self.embedding = layers.Embedding(vocab_size, embedding_dim,
                                          input_length=maxlen,
                                          weights=[embedding_matrix],
                                          trainable=False)

        self.bigram = layers.Conv1D(filters=nb_filters, kernel_size=2,
                                    padding="valid", activation="relu")
        self.trigram = layers.Conv1D(filters=nb_filters, kernel_size=3,
                                    padding="valid", activation="relu")
        self.fourgram = layers.Conv1D(filters=nb_filters, kernel_size=4,
                                    padding="valid", activation="relu")
        self.pool1 = layers.GlobalMaxPooling1D()
        self.pool2 = layers.GlobalMaxPooling1D()
        self.pool3 = layers.GlobalMaxPooling1D()
        self.dense_1 = layers.Dense(units=FFN_units, activation="relu")
        self.dropout = layers.Dropout(rate=dropout_rate)
        if nb_classes == 2:
            self.last_dense = layers.Dense(units=1, activation="sigmoid")
        else:
            self.last_dense = layers.Dense(units=nb_classes, activation="softmax")

    def call(self, inputs, training):
        embs = self.embedding(inputs)
        x_1 = self.bigram(embs)
        x_1 = self.pool1(x_1)
        x_2 = self.trigram(embs)
        x_2 = self.pool2(x_2)
        x_3 = self.fourgram(embs)
        x_3 = self.pool3(x_3)
        merged = tf.concat([x_1, x_2, x_3], axis=-1) # (batch_size, 3*nb_filters)
        merged = self.dense_1(merged)
        merged = self.dropout(merged, training)
        output = self.last_dense(merged)

        return output

    # AFAIK: The most convenient method to print model.summary() 
    # similar to the sequential or functional API like.
    def build_graph(self, input_shape):
        x = Input(shape=input_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x, False))

NB_FILTER = 100
FFN_UNITS = 256
NB_CLASSES = 5

DROPOUT_RATE = 0.2

batch_size = 64
epochs = 100



dcnn = CNN234GramModel(nb_filters=NB_FILTER,
            FFN_units=FFN_UNITS, nb_classes=NB_CLASSES,
            dropout_rate=DROPOUT_RATE)

if NB_CLASSES == 2:
    dcnn.compile(loss="binary_crossentropy",
                 optimizer="adam", metrics=["accuracy"])
else:
    dcnn.compile(loss="categorical_crossentropy", optimizer="adam",
                 metrics=["accuracy"])
  
def training(model):
    history = model.fit(X_train, y_train, 
                        batch_size=batch_size, epochs=epochs, 
                        class_weight=d_class_weights, shuffle=True,
                        verbose=1, validation_split=0.1)

    plot_history(history)

    loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))

    predict('I liked the product quite a lot.', model, tokenizer)
    predict('playing this was the worst time in my life.', model, tokenizer)

    y_pred = model.predict(X_test)

    print('accuracy:', accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)))
    print('matthews_corrcoef:', matthews_corrcoef(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)))
    print('classification_report:\n', classification_report_imbalanced(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)))

input_shape = [batch_size, 100]
dcnn.build(input_shape)
# dcnn.build_graph(input_shape).summary()
dcnn.summary()

"""### training the v1 model"""

training(dcnn)

"""### plot of model metrics till now"""

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns

sns.set(rc={'figure.figsize':(11.7,8.27)})

# data
x = ["CV+NB",             "FF",               "CNN_trigram",      "CNN234GramModel"]
y = [0.33504969057289086, 0.2148927270393169, 0.3006712012400788, 0.29459743828288504]

ax = sns.barplot(x=x, y=y)

"""### save the 234-CNN model"""

cnn_234gram_model_path = "/content/drive/MyDrive/educative_natural_language_processing_for_professionals/models/cnn_234gram_model"
dcnn.save(cnn_234gram_model_path)

"""## CNN234GramModel with batch norm"""

import string
import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
plt.style.use('ggplot')


# map punctuation to space
translator = str.maketrans(string.punctuation, ' '*len(string.punctuation)) 

def text_preprocessing(text):
    """
    Preprocess the text for better understanding
    
    """
    text = text.strip()
    text = text.lower()
    text = text.translate(translator)
    return text

def predict(text, model, tokenizer):
    orig_text = text
    text = text_preprocessing(text)
    text = [text]
    X = tokenizer.texts_to_sequences(text)
    X = pad_sequences(X, padding='post', maxlen=maxlen)
    # print(np.sum(model.predict(X)[0]))
    probs = model.predict(X)[0]
    # assert np.sum(probs) == 1.0
    prediction = np.argmax(probs)
    print('input text: ', orig_text)
    print('predicted class: ', prediction + 1, 'confidence:', np.max(probs))


def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    

class CNN234GramModel1(tf.keras.Model):
    def __init__(self, nb_filters=50,
                 FFN_units=512, nb_classes=2, dropout_rate=0.1, name="dcnn"):
        super(CNN234GramModel1, self).__init__(name=name)
        
        self.embedding = layers.Embedding(vocab_size, embedding_dim,
                                        input_length=maxlen,
                                        weights=[embedding_matrix],
                                        trainable=False)

        self.bigram = layers.Conv1D(filters=nb_filters, kernel_size=2, padding="valid")
        self.trigram = layers.Conv1D(filters=nb_filters, kernel_size=3, padding="valid")
        self.fourgram = layers.Conv1D(filters=nb_filters, kernel_size=4, padding="valid")
        self.batchnorm = layers.BatchNormalization()
        self.activation = layers.Activation('relu')
        self.pool1 = layers.GlobalMaxPooling1D()
        self.pool2 = layers.GlobalMaxPooling1D()
        self.pool3 = layers.GlobalMaxPooling1D()
        self.dense_1 = layers.Dense(units=FFN_units, activation="relu")
        self.dropout = layers.Dropout(rate=dropout_rate)
        if nb_classes == 2:
        self.last_dense = layers.Dense(units=1, activation="sigmoid")
        else:
        self.last_dense = layers.Dense(units=nb_classes, activation="softmax")
        
    def convolv(self, ngram, embs):
        x = ngram(embs)
        x = self.batchnorm(x)
        x = self.activation(x)
        x = self.pool(x)
        return x
        
    def call(self, inputs, training):
        embs = self.embedding(inputs)
        x_1 = self.convolv(self.bigram, embs)
        x_2 = self.convolv(self.trigram, embs)
        x_3 = self.convolv(self.fourgram, embs)
        merged = tf.concat([x_1, x_2, x_3], axis=-1) # (batch_size, 3*nb_filters)
        merged = self.dense_1(merged)
        merged = self.dropout(merged, training)
        output = self.last_dense(merged)

        return output

NB_FILTER = 100
FFN_UNITS = 256
NB_CLASSES = 5

DROPOUT_RATE = 0.2

NB_EPOCHS = 5



dcnn_v2 = CNN234GramModel1(nb_filters=NB_FILTER, FFN_units=FFN_UNITS, 
                           nb_classes=NB_CLASSES, dropout_rate=DROPOUT_RATE)

if NB_CLASSES == 2:
    dcnn_v2.compile(loss="binary_crossentropy",
                    optimizer="adam", metrics=["accuracy"])
else:
    dcnn_v2.compile(loss="categorical_crossentropy", optimizer="adam",
                    metrics=["accuracy"])

"""### training the v2 model"""

training(dcnn_v2)

"""### plot the model metrics till now"""

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})

# data
x = ["CV+NB",             "FF",               "CNN_trigram",      "CNN234GramModel",   "CNN234GramModel_v2"]
y = [0.33504969057289086, 0.2148927270393169, 0.3006712012400788, 0.29459743828288504, 0.22974056487796335]

ax = sns.barplot(x=x, y=y)

"""### save the CNN v2 model"""

cnn_234gram_modelv2_path = "/content/drive/MyDrive/educative_natural_language_processing_for_professionals/models/cnn_234gram_modelv2"
dcnn_v2.save(cnn_234gram_modelv2_path)

