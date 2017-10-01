#nlp model is trained and saved

import numpy as np
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
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV

def preprocess_text(message):
    # removess any punctuation
    nopunc = [char for char in message if char not in string.punctuation]

    # forms a string without punctuation
    nopunc = ''.join(nopunc)

    # removes any stopwords and returns the rest as list of words
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

def get_data(csvfile):
    # returns as dataframe
    return pd.read_csv(csvfile, header=0, sep=',')

def setup_model():
    # returns a pipeline of transforms with a final estimator
    return Pipeline([('vec', CountVectorizer(analyzer=preprocess_text)),  # strings to vectors
                     ('tfidf', TfidfTransformer()),  # vectors to weighted TF-IDF vectors
                     ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
                    ])

def test_model(model, X, y):
    # predicts labels for X and report against the expected y
    predictions = model.predict(X)
    print(classification_report(predictions, y))

def save_model(model, filename):
    # saves the model to a file
    pickle.dump(model, open(filename, 'wb'))

def fit_model(model, X, y):

    params = {
        'vec__ngram_range': [(1, 1), (1, 2)],
#        'tfidf__smooth_idf': [True, False],
        'classifier__alpha': [0.1, 1, 10],
        'classifier__fit_prior': [True, False],
    }

    best_model = GridSearchCV(model, param_grid=params, cv=5, n_jobs=-1, verbose=1) #, scoring='f1_samples'
    best_model.fit(X, y)
    print ('best model index =', best_model.best_index_)
    print ('best model params=', best_model.best_params_)
    print ('best model       =', best_model.best_estimator_)
    return best_model

def split_data(df, test_portion):
    # custome split per label to ensure each label is included in the
    # training and test sets
    df_train = pd.DataFrame()
    df_test = pd.DataFrame()

    # find out all unique labels
    unique_labels = df['labels'].unique()

    # split and accumulate data for each label into training and test
    # wrt given test size
    for l in unique_labels:
        df_per_label = df[df['labels'] == l]
        per_label_size = round(df_per_label.shape[0])
        test_indices = np.random.choice(per_label_size, round(per_label_size * test_portion), replace=False)
        train_indices = np.array(list(set(range(per_label_size)) - set(test_indices)))

        df_train = df_train.append(df_per_label.iloc[train_indices])
        df_test = df_test.append(df_per_label.iloc[test_indices])

    # shuffle training data before separating it
    for _ in range(100):
        df_train = shuffle(df_train)

    return df_train['data'], df_test['data'], df_train['labels'], df_test['labels']

def main():
    #messages = get_data('data/training data.csv')
    messages = get_data('data/a-s-data.csv')

    # split data for training ang testing
    msg_train, msg_test, label_train, label_test = split_data(messages, 0.3)

    training_model = setup_model()

    best_model = fit_model(training_model, msg_train, label_train)

    test_model(best_model, msg_test, label_test)

    save_model(best_model.best_estimator_, 'nlp_model.sav')

if __name__ == '__main__':
    main()