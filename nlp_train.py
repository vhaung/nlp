#nlp model is trained and saved

import pandas as pd
import string
import pickle

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline


def preprocess_text(message):
    # removess any punctuation
    nopunc = [char for char in message if char not in string.punctuation]

    # forms a string without punctuation
    nopunc = ''.join(nopunc)

    # removes any stopwords and returns the rest as list of words
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

def convert_word2vec(words):
    # returns as bag of words
    return CountVectorizer(analyzer=preprocess_text).fit(words)

def get_data(csvfile):
    # returns as dataframe
    return pd.read_csv(csvfile, header=0, sep=',')

def setup_model():
    # returns a pipeline of transforms with a final estimator
    return Pipeline([('bow', CountVectorizer(analyzer=preprocess_text)),  # strings to bow
                     ('tfidf', TfidfTransformer()),  # bow to weighted TF-IDF vectors
                     ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
                    ])

def test_model(model, X, y):
    # predicts labels for X and report against the expected y
    predictions = model.predict(X)
    print(classification_report(predictions, y))

def save_model(model, filename):
    # saves the model to a file
    pickle.dump(model, open(filename, 'wb'))

def main():
    messages = get_data('data/a-s-data.csv')

    # split data for training ang testing
    msg_train, msg_test, label_train, label_test = train_test_split(messages['data'],
                                                                    messages['labels'],
                                                                    test_size=0.3)

    training_model = setup_model()

    training_model.fit(msg_train,label_train)

    test_model(training_model, msg_test, label_test)

    save_model(training_model, 'nlp_model.sav')

if __name__ == '__main__':
    main()