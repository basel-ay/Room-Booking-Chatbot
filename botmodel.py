import os
import json
import random
import numpy as np
import pandas as pd
from string import punctuation
from nltk import word_tokenize
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


class BotModel(object):
    '''   This is the main bot model class   '''

    def __init__(self):
        '''   sets all constants and file paths   '''

        self.RANDOM_SEED = 123
        self.embeddings_size = 100
        self.train_file_path = './data/generated_train_data.json'
        self.bot_clf_path = './models/bot_clf.joblib'
        self.lbencoder_path = './models/lbencoder.joblib'
        self.tfidf_path = './models/tfidf.joblib'
        self.glove_path = './models/glove.6B.%dd.txt' % self.embeddings_size

        self.embeddings_index = None
        self.tfidf = None
        self.tfidf_dict = None
        self.bot_clf = None
        self.lbencoder = None
        self.initialized = False

        # For randomization and re-producability of results
        random.seed(self.RANDOM_SEED)
        np.random.seed(self.RANDOM_SEED)

    def initialize(self):
        '''   loads the Bot models and pre-requisites. Has to be called before using Bot   '''

        assert os.path.isfile(self.tfidf_path), "TfIdf model file not found"
        assert os.path.isfile(self.bot_clf_path), "Bot model file not found"
        assert os.path.isfile(
            self.lbencoder_path), "LabelEncoder model file not found"
        assert os.path.isfile(
            self.train_file_path), "Training data file not found"

        self.embeddings_index = self.__load_glove_vectors()
        self.tfidf = load(self.tfidf_path)
        self.lbencoder = load(self.lbencoder_path)
        self.bot_clf = load(self.bot_clf_path)
        self.tfidf_dict = dict(
            zip(self.tfidf.get_feature_names(), list(self.tfidf.idf_)))
        training_data = self.__load_train_data()
        self.query_response = {td['intent']: td['response']
                               for td in training_data}
        self.initialized = True
        print("Bot Loaded..")

    def __load_glove_vectors(self):
        '''   Loads the glove vectors required for featurization   '''

        embeddings_index = dict()

        if not os.path.isfile(self.glove_path):
            raise Exception("Glove vectors file not found: %s" %
                            self.glove_path)

        with open(self.glove_path, encoding='cp437') as gfile:
            for line in gfile:
                values = line.split()
                word, vectors = values[0], np.asarray(
                    values[1:], dtype='float32')
                embeddings_index[word] = vectors
        return embeddings_index

    def __load_train_data(self):
        '''   Loads the training data for training   '''

        training_data = None
        if not os.path.isfile(self.train_file_path):
            raise Exception("Training data file not found: %s" %
                            self.train_file_path)

        with open(self.train_file_path, 'r') as file:
            training_data = json.load(file)
        return training_data

    def __get_tfIdf_weighted_glove_vectors(self, queries):
        '''   returns Tf-Idf weighted average word vectors(Glove)   '''

        tfidf_weighted_glove = []

        for query in queries:
            tokens = [tokn.lower() for tokn in word_tokenize(query)
                      if tokn not in list(punctuation)]
            query_vec = np.zeros(self.embeddings_size)
            weight_sum = 0
            for tokn in tokens:
                if tokn in self.embeddings_index and tokn in self.tfidf_dict:
                    vec = self.embeddings_index[tokn]
                    # the tf-Idf score of a word in query is pumped up based on the ratio of its
                    # count in the query to the total query length
                    score = self.tfidf_dict[tokn] * \
                        ((tokens.count(tokn)/len(tokens))+1)
                    query_vec += (vec * score)
                    weight_sum += score
                else:
                    pass

            if weight_sum != 0:
                query_vec /= weight_sum
            tfidf_weighted_glove.append(query_vec)
        tfidf_weighted_glove = np.array(tfidf_weighted_glove)
        return tfidf_weighted_glove

    def __get_query_features(self, queries):
        '''   returns concatenated Tf-Idf features and the Tf-Idf weighted average Glove   '''

        tfidf_weighted_glove = self.__get_tfIdf_weighted_glove_vectors(queries)
        tfidf_features = self.tfidf.transform(queries).todense()

        return np.hstack((tfidf_features, tfidf_weighted_glove))

    def train(self, save_models=False):
        '''   Trains the bot model afresh   '''

        try:
            if not self.initialized:
                # Load the Glove Vectors if not initialized
                self.embeddings_index = self.__load_glove_vectors()

            training_data = self.__load_train_data()

            # Read the intents and queries
            queries, intents = [], []
            for train_set in training_data:
                for query in train_set['query']:
                    queries.append(query)
                    intents.append(train_set['intent'])

            # Separate the data for train(&cv) and test
            queries_train, queries_test, intents_train, intents_test = train_test_split(queries,
                                                                                        intents, train_size=0.9, random_state=self.RANDOM_SEED, stratify=intents)

            # Setup Tf-Idf Vectorizer and fit on training data
            self.tfidf = TfidfVectorizer(max_features=600, encoding='latin-1', sublinear_tf=True,
                                         lowercase=True, tokenizer=word_tokenize, ngram_range=(1, 2),
                                         stop_words=list(punctuation), token_pattern=None)
            self.tfidf.fit(queries_train)

            # Tf-Idf feature-score mapping
            self.tfidf_dict = dict(
                zip(self.tfidf.get_feature_names(), list(self.tfidf.idf_)))
            # tfidf_feat = tfidf.get_feature_names()

            # Get the complete Query features for Train and Test
            X_train = self.__get_query_features(queries_train)
            X_test = self.__get_query_features(queries_test)

            # Set up and fit Label Encoder
            self.lbencoder = LabelEncoder()
            self.lbencoder.fit(intents_train)

            # Get the class labels for Train and Test
            Y_train = self.lbencoder.transform(intents_train)
            Y_test = self.lbencoder.transform(intents_test)

            # Define the classifier
            self.bot_clf = LogisticRegression(C=1, penalty='l2', solver='newton-cg',
                                              random_state=self.RANDOM_SEED, n_jobs=-1)
            self.bot_clf.fit(X_train, Y_train)

            print("Train accuracy : %.3f" %
                  (accuracy_score(Y_train, self.bot_clf.predict(X_train))))
            Y_pred = self.bot_clf.predict(X_test)
            print("Test accuracy : %.3f" % (accuracy_score(Y_test, Y_pred)))
            print("F1 Score : %.3f" %
                  (f1_score(Y_test, Y_pred, average='weighted')))

            if save_models:
                dump(self.bot_clf, self.bot_clf_path)
                dump(self.tfidf, self.tfidf_path)
                dump(self.lbencoder, self.lbencoder_path)

        except Exception as ex:
            print("Error in Bot Training..!!: %s" % str(ex))

    def predict(self, query):
        '''   function for debugging on the trained model   '''

        query_features = self.__get_query_features([query])
        pred = self.bot_clf.predict_proba(query_features)
        tag = self.lbencoder.inverse_transform([pred.argmax()])[0]
        conf = pred[0][pred.argmax()]
        return tag, conf

    def response(self, query):
        '''   function for using the trained saved model for direct prediction   '''
        try:
            if not self.initialized:
                raise Exception(
                    "First initialize the BotModel by running .initialize() method")

            query_features = self.__get_query_features([query])
            pred = self.bot_clf.predict_proba(query_features)
            tag = self.lbencoder.inverse_transform([pred.argmax()])[0]
            conf = pred[0][pred.argmax()]
            resp = random.choice(self.query_response[tag])
            return tag, conf, [resp]
        except Exception as ex:
            print("Bot Error : %s" % str(ex))
            return


if __name__ == "__main__":
    model = BotModel()
    model.train(save_models=True)
    model.initialize()
