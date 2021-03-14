import json,re, time
import keras
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.model_selection import train_test_split   
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv1D, Conv2D, Flatten, LSTM
from keras.layers.embeddings import Embedding
import pandas as pd
from keras.layers.convolutional import MaxPooling1D
from sklearn.metrics import classification_report
from w2vposts import GensimW2V
from CleanDataClass import CleanDataClass
import tensorflow
import readcorpus as rc
from features import getFeatures, getFeatureschi2
from sklearn.preprocessing import label_binarize
import csv

class CnnFeature(object):
    """docstring for CNNLearnClass"""

    def __init__(self, corpus=1, clas='Polarity Class'):
        """constructor."""
        print('corpus:', corpus)
        self.corpus = corpus
        self.clas = clas

        self.input_data = rc.set_input_data(None, corpus, clas=clas)
        self.output_data = rc.set_output_data(None, corpus, clas=clas)
        self.c, self.listclasses = rc.getdictclasses(self.output_data)
        self.nb_classes = rc.getnumberclasses(self.output_data)
        self.c_list = list(self.c.values())

    def getSubWords(self, tweet):
        x = tweet.split(' ')
        r = []
        for word in x:
            c = set(
                word[i:j]
                for i in range(len(word)) for j in range(i + 3, len(word) + 1)
            )
            r.extend(c)
        return ' '.join(r)

    def convert_text_to_index_array(self, text, dictionary=None):
        if dictionary is None:
            dictionary = self.dictionary
        return [dictionary[word] for word in kpt.text_to_word_sequence(text)]

    def splitData(self, thedata, algo):
        if algo == 'cnnwe':
            thedata = thedata
            emb_size = 300
        elif algo in ['sub']:
            emb_size = 100
            thedata = [self.getSubWords(x) for x in thedata]
        elif algo == 'char':
            thedata = [' '.join(list(x)) for x in thedata]
            emb_size = 15
        return thedata, emb_size


    def cnn(self, algo, max_words=3000, feats=False, chi2=False):
        print('type == ', algo, feats, 'chi2=', str(chi2), max_words)
        thedata, emb_size = self.splitData(self.input_data, algo)

        if self.clas in [5, 8, 9]:
            testset = rc.set_input_data(None, (self.clas*10+1), clas=self.clas)
            testset, emb_size = self.splitData(testset, algo)

        print('input len:', len(thedata))
        tokenizer = Tokenizer(num_words=max_words)
        tokenizer.fit_on_texts(thedata)
        self.dictionary = tokenizer.word_index
        vocab_size = len(tokenizer.word_index) + 1

        allWordIndices = []
        for text in thedata:
            wordIndices = self.convert_text_to_index_array(text)
            allWordIndices.append(wordIndices)

        allWordIndices = np.asarray(allWordIndices)


        mode = ["binary"]
        for m in mode:
            print('mode', m)
            train_x = tokenizer.sequences_to_matrix(allWordIndices, mode=m)

            if feats:
                if chi2:
                    featus = getFeatureschi2(self.corpus, clas=self.clas)
                else:
                    featus = getFeatures(self.corpus, clas=self.clas)

                print('Stats::', featus.shape)
                print('Stats::', train_x.shape)
                train_x = np.hstack((train_x, featus))
                print('Stats::', featus.shape)
                print('Stats::', train_x.shape)

            train_y = list(map(lambda x: self.c[x], self.output_data))
            train_y = keras.utils.to_categorical(train_y, self.nb_classes)

            X_train, X_test, Y_train, Y_test = train_test_split(train_x, train_y, test_size = 0.2, shuffle=True)
            input_size = len(train_x[0])
            # one test
            validation_split = [0.1]
            batch = [200]
            nb_neurone = [50]  # number of filters
            nb_epoch = [5]
            activation = ['relu']
            optimizer = ['adam']
            loss = ['mse']
            
            model = Sequential()
            print('emb_size', emb_size)
            model.add(Embedding(vocab_size, emb_size, input_length=input_size))
            model.add(Conv1D(nb, activation=a, kernel_size=self.nb_classes, input_shape=(input_size, 1)))

            model.add(MaxPooling1D(self.nb_classes))
            model.add(Flatten())

            model.add(Dense(self.nb_classes, activation='sigmoid'))
            model.compile(
                loss=l,
                optimizer=o,
                metrics=['accuracy'])

            model.fit(
                X_train,
                Y_train,
                batch_size=b,
                epochs=epoch,
                verbose=1,
                validation_split=vs
            )
            print('evaluation')
            y_pred = model.predict(X_test)

            score = model.evaluate(X_test, Y_test, verbose=0)
            print('acc', score[1])
            Yt_test = np.argmax(Y_test, axis=1) # Convert one-hot to index
            y_pred = model.predict_classes(X_test)
            print(classification_report(Yt_test, y_pred, digits=4))
            print('done')
