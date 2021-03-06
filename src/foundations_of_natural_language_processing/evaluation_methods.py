# -*- coding: utf-8 -*-
"""evaluation_methods

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1tXDJQapwDLTI2zcZZ60La271PdnFVSAe

## download the data

`glove.6B.zip` and `Video_Games_5.json.gz` files are downloaded from the respective url locations. These are done using python functions.
"""

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

# map punctuation to space
translator = str.maketrans(string.punctuation, ' '*len(string.punctuation)) 

def text_preprocessing(text):
    """
    Preprocess the text for better understanding
    
    """
    if isinstance(text, str):
        text = text.strip()
        text = text.lower()
    return text


Video_Games_5 = wget('http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Video_Games_5.json.gz')
df = pd.read_json("./Video_Games_5.json.gz", lines=True, compression='gzip')
df = df[['reviewText', 'overall']]
df['reviewText'] = df.reviewText.apply(text_preprocessing)
df = df.dropna()
df = df.drop_duplicates()
print(df.shape)

"""## train test split

spplit the dataset based on the label distribution. The test size is 0.3 and random state is given so that the split is the same for the different lessons.
"""

from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df, test_size=0.3, stratify=df.overall, random_state=42)

X_train = df_train['reviewText']
y_train = df_train['overall']

X_test = df_test['reviewText']
y_test = df_test['overall']

print(len(df_train), len(df_test))

"""## baseline classifier

A very naive classifier where the labels are assigned randomly.

The precision values for the baseline classifier are according to the distribution of the output class and the recall is 20%. The  matthews correlation coefficient is almost 0 which means that the  classifier has not information on the output label. This is obvious now since we know the model, but sometimes a neural network may also give such values. This is the sign that neural network is also nothing better than a random classfier.
"""

import random

random.seed(42)

def baseline_classifier(text):
    """
    Baseline classifier returning a label randomly
    """
    return float(random.choice([1, 2, 3, 4, 5]))

predictions = df_test['reviewText'].apply(baseline_classifier)

from sklearn.metrics import accuracy_score, matthews_corrcoef
from imblearn.metrics import classification_report_imbalanced

print('accuracy from scikitlearn:', accuracy_score(df_test['overall'], predictions))
print('matthews_corrcoef:', matthews_corrcoef(df_test['overall'], predictions))
print('classification_report:\n', classification_report_imbalanced(df_test['overall'], predictions))

"""## Custom accuracy function"""

def calculate_acc(gold_labels, preds):
    assert len(gold_labels) == len(preds), "gold labels and predictions should have the same number of values"
    total_values = len(gold_labels)
    matching = sum(int(g==p) for g, p in zip(gold_labels, preds))
    return matching / total_values

print('my accuracy function output', calculate_acc(df_test['overall'].values.tolist(), predictions.values.tolist()))

"""## precision score"""

from sklearn.metrics import precision_score

print('scikitlearn precision score:', precision_score(
    df_test['overall'].values.tolist(), predictions.values.tolist(),
    labels=[1], average=None))

def calculate_precision(gold_labels, preds, target_label):
    assert len(gold_labels) == len(preds), "gold labels and predictions should have the same number of values"
    true_positives = sum(int(g==p==target_label) for g, p in zip(gold_labels, preds))
    all_positive_predictions = sum(int(p==target_label) for p in preds)
    return true_positives / all_positive_predictions

print('my precision functions', calculate_precision(
    df_test['overall'].values.tolist(), predictions.values.tolist(), 1))

"""## recall score"""

from sklearn.metrics import recall_score

print('scikitlearn recall score:', recall_score(
    df_test['overall'].values.tolist(), predictions.values.tolist(),
    labels=[1], average=None))

def calculate_recall(gold_labels, preds, target_label):
    assert len(gold_labels) == len(preds), "gold labels and predictions should have the same number of values"
    true_positives = sum(int(g==p==target_label) for g, p in zip(gold_labels, preds))
    all_correct_samples = sum(int(g==target_label) for g in gold_labels)
    return true_positives / all_correct_samples

print('my recall functions', calculate_recall(
    df_test['overall'].values.tolist(), predictions.values.tolist(), 1))

"""## f1 score"""

from sklearn.metrics import f1_score

print('scikitlearn f1 score:', f1_score(
    df_test['overall'].values.tolist(), predictions.values.tolist(),
    labels=[1], average=None))

print('manual f1score calculation:', 2 * 0.06834353797782558 * 0.19699499165275458 / (0.06834353797782558 + 0.19699499165275458))

"""## classification report"""

from imblearn.metrics import classification_report_imbalanced

print(classification_report_imbalanced(df_test['overall'], predictions))

"""## confusion matrix

the output as per the scikit learn documentation

Confusion matrix whose i-th row and j-th column entry indicates the number of samples with true label being i-th class and predicted label being j-th class.

annot=True to annotate cells, ftm='g' to disable scientific notation
"""

# Commented out IPython magic to ensure Python compatibility.
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
# %matplotlib inline

def plot_confusion_mat(cf_mat, labels):
    plt.figure(figsize=(10, 8))
    ax = plt.subplot()
    sn.heatmap(cf_mat, annot=True, fmt='g', ax=ax, xticklabels=labels, yticklabels=labels)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    plt.show()


labels = [1,2,3,4,5]
cf_mat = confusion_matrix(df_test['overall'], predictions, labels=labels)
plot_confusion_mat(cf_mat, labels)

# Commented out IPython magic to ensure Python compatibility.
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
# %matplotlib inline


labels = [1,2,3,4,5]
cf_mat = confusion_matrix(df_test['overall'].values.tolist(), df_test['overall'].values.tolist(), labels=labels)
plot_confusion_mat(cf_mat, labels)

"""## matthews correlation coefficient"""

from sklearn.metrics import matthews_corrcoef

print('MCC for random output:', matthews_corrcoef(
    df_test['overall'], predictions))
print('MCC where only one class is predicted:', matthews_corrcoef(
    df_test['overall'], [1] * len(df_test['overall'])))
print('MCC where all classes are correctly predicted:', matthews_corrcoef(
    df_test['overall'], df_test['overall']))
print('MCC where all classes are incorrectly predicted:', matthews_corrcoef(
    df_test['overall'], [(val + 1) % 6 for val in df_test['overall'].values.tolist()]))