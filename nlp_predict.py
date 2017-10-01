#load the saved nlp model to predict text labels

import pandas as pd
import string
import pickle
import sys

from nltk.corpus import stopwords
from sklearn.metrics import classification_report

# model must be loaded before doing any prediction with it
loaded_model = {}

# The same preprocess_text() function used during model training is provided here.
# Looks like it's called as part of the pipeline process to predict label.
def preprocess_text(message):
    # removes any punctuation
    nopunc = [char for char in message if char not in string.punctuation]

    # forms a string without punctuation
    nopunc = ''.join(nopunc)

    # removes any stopwords and returns the rest as list of words
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

def get_data(csvfile):
    # returns as dataframe
    return pd.read_csv(csvfile, header=0, sep=',')

def load_model(filename):
    # loads the saved model from the file
    global loaded_model
    loaded_model = pickle.load(open(filename, 'rb'))

def test_model(X, y):
    # predicts labels for X and report against the expected y
    predictions = predict_label(X)
    print(classification_report(predictions, y))

def predict_label(X):
    # predicts labels for X using the loaded model
    if loaded_model == {}:
        sys.exit('error: load the model first')
    else:
        return loaded_model.predict(X)

def main():
    # load the model from a saved file for testing purpose
    load_model('nlp_model.sav')

    # get data for testing purpose
    messages = get_data('data/a-s-data.csv')
    #messages = get_data('data/training data.csv')

    # testing the loaded model using all data
    test_model(messages['data'], messages['labels'])

if __name__ == '__main__':
    main()
