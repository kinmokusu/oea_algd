#!/usr/bin/env python
# coding: utf-8
from sklearn.preprocessing import label_binarize
from CleanDataClass import CleanDataClass
import keras.preprocessing.text as kpt
import keras
from keras.preprocessing.text import Tokenizer
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import json
from sklearn.feature_extraction.text import CountVectorizer
import readcorpus as rc
import csv


# create training + testing sets
def convert_text_to_index_array(text, dictionary):
    # one really important thing that `text_to_word_sequence` does
    # is make all texts the same length -- in this case, the length
    # of the longest text in the set.
    return [dictionary[word] for word in kpt.text_to_word_sequence(text)]

def getnumberclasses(y):
    """Return number classes."""
    return len(set(y))


def getdictclasses(y):
    """Return the dictionary of all classes."""
    classes = set(y)
    i = 0
    result = {}
    for c in classes:
        result[c] = i
        i += 1
    return result, classes
##############################################################
# get the datasets
sets = [1]

for corpus in sets:
    clas = corpus
    print('corpus:', corpus)
    input_data = rc.set_input_data(None, corpus, clas=clas)
    output_data = rc.set_output_data(None, corpus, clas=clas)

    print('len(input_data)', len(input_data))
    print('len(output_data)', len(output_data))

    nb_classes = getnumberclasses(output_data)
    c, listclasses = getdictclasses(output_data)
    c_list = list(c.values())
    print('c = ', c)

    nbwords = 3000

    max_words = nbwords

    # create a new Tokenizer
    tokenizer = Tokenizer(num_words=max_words)
    # feed our tweets to the Tokenizer
    tokenizer.fit_on_texts(input_data)

    dictionary = tokenizer.word_index

    allWordIndices = []
    # for each tweet, change each token to its ID in the Tokenizer's word_index
    for text in input_data:
        wordIndices = convert_text_to_index_array(text, dictionary)
        allWordIndices.append(wordIndices)
    allWordIndices = np.asarray(allWordIndices)

    best_precision = 0
    # experiment with different text representations
    modes = ['freq', 'binary', 'count', 'tfidf']
    for mode in modes:
        X = tokenizer.sequences_to_matrix(allWordIndices, mode=mode)
        y = list(map(lambda x: c[x], output_data))
        y = keras.utils.to_categorical(y, nb_classes)

        Y = label_binarize(y, classes=c_list)
        print('c_list', c_list)
        n_classes = Y.shape[1]
        test_size = [0.1]
        for ts in test_size:
            print ('ts ', ts)
            # Split into training and test
            random_state = np.random.RandomState(0)
            for rs in [0, 1]:
                if rs:
                    print('rs')
                    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=ts, random_state=random_state, stratify=Y)
                else:
                    print('not rs')
                    if clas in [5, 8, 9]:
                        X_train = X
                        X_test = tokenizer.sequences_to_matrix(allWordIndices, mode=mode)
                        Y_train = Y
                        Y_test = rc.set_output_data(None, (clas*10+1), clas=clas)
                        Y_test = label_binarize(Y_test, classes=c_list)
                        Y_test = keras.utils.to_categorical(Y_test, nb_classes)
                    else:
                        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=ts)

                # Run classifier
                cl = ['svm', 'svc', 'bisvc']

                for c in cl:
                    print('classifier::', c)

                    if c == 'bisvc':
                        svcbi()
                        pass

                    if c == 'svm':
                        print('classifier svm')
                        classifier = OneVsRestClassifier(svm.LinearSVR())
                    else:
                        print('classifier else')
                        classifier = OneVsRestClassifier(svm.LinearSVC())

                    print('fitting')
                    classifier.fit(X_train, Y_train)

                    y_score = classifier.predict(X_test)
                    print(classification_report(Y_test, y_score, digits=4))
                    print ('acc', accuracy_score(Y_test, y_score))

                    print('error analysis')
                    for x in range(0, len(y_score)):
                        s = len(Y_train)
                        if not np.array_equal(y_score[x], Y_test[x]):
                            print('Row', x, 'has been classified as ', y_score[x], 'and should be ', Y_test[x], 'Data:', input_data[s + x])

                    print('end error analysis!!!')
print('done')
