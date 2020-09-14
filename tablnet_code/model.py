#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
reproduce the table alignment modules of TableNet 
LSTM, BiLSTM, BiLSTM with column-attention
"""

# %%
import gzip
import json
import os
import numpy as np
import pickle
from operator import itemgetter
from gensim.models import KeyedVectors
from nltk.corpus import stopwords

# %%
import keras
from keras import Model, Input
from keras.layers import Activation, Bidirectional, Dense, Dropout, Embedding, Flatten, Lambda, LSTM, Input, merge, TimeDistributed, RepeatVector, Permute, Reshape
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import *
from keras.models import *
from keras.optimizers import *
from keras.utils.np_utils import to_categorical
from keras.layers.core import *
from keras import backend as K
from keras import metrics

# %%
from os import listdir
from os.path import isfile, join
from sklearn.metrics import confusion_matrix as cm

import pandas as pd
from pandas import DataFrame
from pandas.io.json import json_normalize
pd.set_option('display.max_colwidth', 50)

import random
# from DataLoader import DataLoader
from table import WikiTable

import sys
sys.setrecursionlimit(100000)

# %%
'''Compute P/R/F1 from the confusion matrix. '''
def evaluation_metrics_report(mat, labels_, method_, epochs=10):
    num_classes = len(mat)
    scores = dict()
    avg_p = []
    avg_r = []
    avg_f1 = []
    for i in range(0, num_classes):
        p = mat[i,i] / float(sum(mat[:,i]))
        r = mat[i,i] / float(sum(mat[i,:]))
        f1 = 2 * (p * r) / (p + r)
        scores[i] = (p, r, f1)
        avg_p.append(p)
        avg_r.append(r)
        avg_f1.append(f1)
    outstr = 'Evaluation results for ' + method_ + ' Epochs: ' + str(epochs) + '\n'
    for key in scores:
        label = labels_[key]
        val_1 = scores[key][0]
        val_2 = scores[key][1]
        val_3 = scores[key][2]
        outstr += ('%s\tP=%.3f\tR=%.3f\tF1=%.3f\n' % (label, val_1, val_2, val_3))
    avg_p_score = sum(avg_p) / len(avg_p)
    avg_r_score = sum(avg_r) / len(avg_r)
    avg_f1_score = sum(avg_f1) / len(avg_f1)
    outstr += 'AVG\tAvg-P=%.3f\tAvg-R=%.3f\tAvg-F1=%.3f\n' % (avg_p_score, avg_r_score, avg_f1_score)
    return outstr

# %% LSTM column-by-column with attention model

def get_H_n(X):
    ans = X[:, -1, :]  # get last element from time dim
    return ans


def get_Y(X, xmaxlen):
    return X[:, :xmaxlen, :]  # get first xmaxlen elem from time dim


def get_R(X):
    Y, alpha = X[0], X[1]
    ans = K.batch_dot(Y, alpha)
    return ans

# %%
'''LSTM baseline where the columns are represented by their title, and LCA category. '''

def build_lstm_baseline_w2v(w2v_size, w2v_dim, word2vec_matrix, lstm_units=100, col_length=10):
    k = 2 * lstm_units
    N = col_length

    # the first layer is the embeddings for the column names
    main_input = Input(shape=(N,), dtype='int32', name='main_input')
    e = Embedding(w2v_size, w2v_dim, weights=[word2vec_matrix], input_length=N, trainable=False)(main_input)

    # merge the two embedding layers
    drop_out = Dropout(0.3, name='dropout')(e)
    # pass the input through a BiLSTM model through the previous layer
    lstm = LSTM(lstm_units, return_sequences=True)(drop_out)
    # add another drop-out layer
    drop_out = Dropout(0.2)(lstm)
    flatten = Flatten()(drop_out)
    out = Dense(3, activation='softmax')(flatten)

    model = Model(input=main_input, output=out)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['acc'])
    return model


'''LSTM baseline where the columns are represented by their title, and LCA category. '''
def build_lstm_baseline_w2v_lca(w2v_size, w2v_dim, word2vec_matrix, n2v_size, n2v_dim, node2vec_matrix, lstm_units=100, col_length=10):
    k = 2 * lstm_units
    N = col_length

    # the first layer is the embeddings for the column names
    main_input = Input(shape=(N,), dtype='int32', name='main_input')
    e = Embedding(w2v_size, w2v_dim, weights=[word2vec_matrix], input_length=N, trainable=False)(main_input)

    # add the second input layer which corresponds to the category embeddings
    subject_input = Input(shape=(N,), dtype='int32', name='subject_input')
    e2 = Embedding(n2v_size, n2v_dim, weights=[node2vec_matrix], input_length=N, trainable=False)(subject_input)

    # merge the two embedding layers
    e4 = merge([e, e2], mode='sum')

    # merge the two embedding layers
    drop_out = Dropout(0.3, name='dropout')(e4)
    # pass the input through a BiLSTM model through the previous layer
    lstm = LSTM(lstm_units, return_sequences=True)(drop_out)
    # add another drop-out layer
    drop_out = Dropout(0.2)(lstm)
    flatten = Flatten()(drop_out)
    out = Dense(3, activation='softmax')(flatten)

    model = Model(input=[main_input, subject_input], output=out)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['acc'])
    return model


'''LSTM baseline where the columns are represented by their title, LCA category, and column values. '''
def build_lstm_baseline_w2v_lca_val(w2v_size, w2v_dim, word2vec_matrix, n2v_size, n2v_dim, node2vec_matrix, lstm_units=100, col_length=10):
    k = 2 * lstm_units
    N = col_length

    # the first layer is the embeddings for the column names
    main_input = Input(shape=(N,), dtype='int32', name='main_input')
    e = Embedding(w2v_size, w2v_dim, weights=[word2vec_matrix], input_length=N, trainable=False)(main_input)

    # add the second input layer which corresponds to the category embeddings
    subject_input = Input(shape=(N,), dtype='int32', name='subject_input')
    e2 = Embedding(n2v_size, n2v_dim, weights=[node2vec_matrix], input_length=N, trainable=False)(subject_input)

    # add the second input layer which corresponds to the category embeddings
    value_input = Input(shape=(N,), dtype='int32', name='value_input')
    e3 = Embedding(n2v_size, n2v_dim, weights=[node2vec_matrix], input_length=N, trainable=False)(value_input)

    # merge the two embedding layers
    e4 = merge([e, e2, e3], mode='sum')

    # merge the two embedding layers
    drop_out = Dropout(0.3, name='dropout')(e4)
    # pass the input through a BiLSTM model through the previous layer
    lstm = LSTM(lstm_units, return_sequences=True)(drop_out)
    # add another drop-out layer
    drop_out = Dropout(0.2)(lstm)
    flatten = Flatten()(drop_out)
    out = Dense(3, activation='softmax')(flatten)

    model = Model(input=[main_input, subject_input, value_input], output=out)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['acc'])
    return model



# %%
'''BiLSTM baseline where the columns are represented by their title. '''
def build_bilstm_baseline_w2v(w2v_size, w2v_dim, word2vec_matrix, lstm_units=100, col_length=10):
    k = 2 * lstm_units
    N = col_length

    # the first layer is the embeddings for the column names
    main_input = Input(shape=(N,), dtype='int32', name='main_input')
    e = Embedding(w2v_size, w2v_dim, weights=[word2vec_matrix], input_length=N, trainable=False)(main_input)

    # merge the two embedding layers
    drop_out = Dropout(0.3, name='dropout')(e)
    # pass the input through a BiLSTM model through the previous layer
    bilstm = Bidirectional(LSTM(lstm_units, return_sequences=True))(drop_out)

    # add another drop-out layer
    drop_out = Dropout(0.2)(bilstm)
    flatten = Flatten()(drop_out)
    out = Dense(3, activation='softmax')(flatten)
    output = out

    model = Model(input=main_input, output=output)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['acc'])
    return model


'''BiLSTM baseline where the columns are represented by their title, and LCA category. '''
def build_bilstm_baseline_w2v_lca(w2v_size, w2v_dim, word2vec_matrix, n2v_size, n2v_dim, node2vec_matrix, lstm_units=100, col_length=10):
    k = 2 * lstm_units
    N = col_length

    # the first layer is the embeddings for the column names
    main_input = Input(shape=(N,), dtype='int32', name='main_input')
    e = Embedding(w2v_size, w2v_dim, weights=[word2vec_matrix], input_length=N, trainable=False)(main_input)

    # add the second input layer which corresponds to the category embeddings
    subject_input = Input(shape=(N,), dtype='int32', name='subject_input')
    e2 = Embedding(n2v_size, n2v_dim, weights=[node2vec_matrix], input_length=N, trainable=False)(subject_input)

    # merge the two embedding layers
    e4 = merge([e, e2], mode='sum')

    # merge the two embedding layers
    drop_out = Dropout(0.3, name='dropout')(e4)
    # pass the input through a BiLSTM model through the previous layer
    bilstm = Bidirectional(LSTM(lstm_units, return_sequences=True))(drop_out)
    # add another drop-out layer
    drop_out = Dropout(0.2)(bilstm)
    flatten = Flatten()(drop_out)
    out = Dense(3, activation='softmax')(flatten)

    model = Model(input=[main_input, subject_input], output=out)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['acc'])
    return model


'''BiLSTM baseline where the columns are represented by their title, LCA category, and column values. '''
def build_bilstm_baseline_w2v_lca_val(w2v_size, w2v_dim, word2vec_matrix, n2v_size, n2v_dim, node2vec_matrix, lstm_units=100, col_length=10):
    k = 2 * lstm_units
    N = col_length

    # the first layer is the embeddings for the column names
    main_input = Input(shape=(N,), dtype='int32', name='main_input')
    e = Embedding(w2v_size, w2v_dim, weights=[word2vec_matrix], input_length=N, trainable=False)(main_input)

    # add the second input layer which corresponds to the category embeddings
    subject_input = Input(shape=(N,), dtype='int32', name='subject_input')
    e2 = Embedding(n2v_size, n2v_dim, weights=[node2vec_matrix], input_length=N, trainable=False)(subject_input)

    # add the second input layer which corresponds to the category embeddings
    value_input = Input(shape=(N,), dtype='int32', name='value_input')
    e3 = Embedding(n2v_size, n2v_dim, weights=[node2vec_matrix], input_length=N, trainable=False)(value_input)

    # merge the two embedding layers
    e4 = merge([e, e2, e3], mode='sum')
    drop_out = Dropout(0.3, name='dropout')(e4)

    # pass the input through a BiLSTM model through the previous layer
    bilstm = Bidirectional(LSTM(lstm_units, return_sequences=True))(drop_out)

    # add another drop-out layer
    drop_out = Dropout(0.2)(bilstm)
    flatten = Flatten()(drop_out)
    out = Dense(3, activation='softmax')(flatten)

    model = Model(input=[main_input, subject_input, value_input], output=out)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['acc'])
    return model



# %%
'''
We build a BiLSTM model with attention, which matches two tables based on their column representation.
The attention mechanism decides which columns have their highest match from the second candidate table
and then uses this to compute the attention weights which later on are used for classifying the pair
as either: 'equivalent', 'subPartOf' or 'notAligned'

The model is based on the paper by Rockta schel et al. "Reasoning about Entailment with Neural Attention"
@Credit: This is an adaptation of the implementation of the original paper by https://github.com/shyamupa/snli-entailment
'''
def build_bilstm_col_model(vocab_size, w2v_dim, embedding_matrix, lstm_units=100, col_length=20):
    k = 2 * lstm_units
    N = col_length
    # the first layer is the embeddings for the column names
    main_input = Input(shape=(N,), dtype='int32', name='main_input')
    e = Embedding(vocab_size, w2v_dim, weights=[embedding_matrix], input_length=N, trainable=False)(main_input)
    drop_out = Dropout(0.1, name='droput')(e)

    # pass the input through a BiLSTM model through the previous layer
    bilstm = Bidirectional(LSTM(lstm_units, return_sequences=True))(drop_out)

    # add another drop-out layer
    drop_out = Dropout(0.1)(bilstm)

    # here we add a custom layer which connects the last input of the first table as the input of the second table
    t2 = Lambda(get_H_n, output_shape=(k,), name='h_n')(drop_out)

    # add the layer which encodes the output labels
    Y = Lambda(get_Y, arguments={'xmaxlen': col_length}, name='Y', output_shape=(col_length, k))(drop_out)
    # add the layer which encodes the weight parameter from the LSTMs cells and the hidden states
    Whn = Dense(2 * lstm_units, W_regularizer=l2(0.01), name="Wh_n")(t2)
    Whn_x_e = RepeatVector(col_length, name="Wh_n_x_e")(Whn)

    # add the attention layer
    WY = TimeDistributed(Dense(2 * lstm_units, W_regularizer=l2(0.01)), name="WY")(Y)
    merged = merge([Whn_x_e, WY], name='merged', mode='sum')
    M = Activation('sigmoid', name='M')(merged)

    alpha_ = TimeDistributed(Dense(1, activation='linear'), name="alpha_")(M)
    flat_alpha = Flatten(name="flat_alpha")(alpha_)
    alpha = Dense(col_length, activation='softmax', name="alpha")(flat_alpha)

    Y_trans = Permute((2, 1), name="y_trans")(Y)  # of shape (None,300,20)
    r_ = merge([Y_trans, alpha], output_shape=(2 * lstm_units, 1), name="r_", mode=get_R)
    r = Reshape((k,), name="r")(r_)

    Wr = Dense(k, W_regularizer=l2(0.01))(r)
    Wh = Dense(k, W_regularizer=l2(0.01))(t2)
    merged = merge([Wr, Wh], mode='sum')
    h_star = Activation('sigmoid')(merged)

    out = Dense(3, activation='softmax')(h_star)
    output = out

    model = Model(input=[main_input], output=output)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['acc'])
    return model


def build_bilstm_col_subject_model(w2v_size, w2v_dim, word2vec_matrix, n2v_size, n2v_dim, node2vec_matrix, lstm_units=100, col_length=10):
    k = 2 * lstm_units
    N = col_length

    # the first layer is the embeddings for the column names
    main_input = Input(shape=(N,), dtype='int32', name='main_input')
    e = Embedding(w2v_size, w2v_dim, weights=[word2vec_matrix], input_length=N, trainable=False)(main_input)

    # add the second input layer which corresponds to the category embeddings
    subject_input = Input(shape=(N,), dtype='int32', name='subject_input')
    e2 = Embedding(n2v_size, n2v_dim, weights=[node2vec_matrix], input_length=N, trainable=False)(subject_input)

    # merge the two embedding layers
    e3 = merge([e, e2], mode='sum')

    drop_out = Dropout(0.3, name='dropout')(e3)

    # pass the input through a BiLSTM model through the previous layer
    bilstm = Bidirectional(LSTM(lstm_units, return_sequences=True))(drop_out)

    # add another drop-out layer
    drop_out = Dropout(0.1)(bilstm)

    # here we add a custom layer which connects the last input of the first table as the input of the second table
    t2 = Lambda(get_H_n, output_shape=(k,), name='h_n')(drop_out)

    # add the layer which encodes the output labels
    Y = Lambda(get_Y, arguments={'xmaxlen': col_length}, name='Y', output_shape=(col_length, k))(drop_out)
    # add the layer which encodes the weight parameter from the LSTMs cells and the hidden states
    Whn = Dense(k, W_regularizer=l2(0.01), name="Wh_n")(t2)
    Whn_x_e = RepeatVector(col_length, name="Wh_n_x_e")(Whn)

    # add the attention layer
    WY = TimeDistributed(Dense(2 * lstm_units, W_regularizer=l2(0.01)), name="WY")(Y)
    merged = merge([Whn_x_e, WY], name='merged', mode='sum')
    M = Activation('tanh', name='M')(merged)

    alpha_ = TimeDistributed(Dense(1, activation='linear'), name="alpha_")(M)
    flat_alpha = Flatten(name="flat_alpha")(alpha_)
    alpha = Dense(col_length, activation='softmax', name="alpha")(flat_alpha)

    Y_trans = Permute((2, 1), name="y_trans")(Y)  # of shape (None,300,20)
    r_ = merge([Y_trans, alpha], output_shape=(k, 1), name="r_", mode=get_R)
    r = Reshape((k,), name="r")(r_)

    Wr = Dense(k, W_regularizer=l2(0.01))(r)
    Wh = Dense(k, W_regularizer=l2(0.01))(t2)
    merged = merge([Wr, Wh], mode='sum')
    h_star = Activation('tanh')(merged)

    out = Dense(3, activation='softmax')(h_star)
    output = out

    model = Model(input=[main_input, subject_input], output=output)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['acc'])
    return model


def build_bilstm_col_subject_val_model(w2v_size, w2v_dim, word2vec_matrix, n2v_size, n2v_dim, node2vec_matrix, lstm_units=100, col_length=10):
    k = 2 * lstm_units
    N = LEN    # ??

    # the first layer is the embeddings for the column names
    main_input = Input(shape=(N,), dtype='int32', name='main_input')
    e = Embedding(w2v_size, w2v_dim, weights=[word2vec_matrix], input_length=N, trainable=False)(main_input)

    # add the second input layer which corresponds to the category embeddings
    subject_input = Input(shape=(N,), dtype='int32', name='subject_input')
    e2 = Embedding(n2v_size, n2v_dim, weights=[node2vec_matrix], input_length=N, trainable=False)(subject_input)

    # add the second input layer which corresponds to the category embeddings
    value_input = Input(shape=(N,), dtype='int32', name='value_input')
    e3 = Embedding(n2v_size, n2v_dim, weights=[node2vec_matrix], input_length=N, trainable=False)(value_input)

    # merge the two embedding layers
    e4 = merge([e, e2, e3], mode='sum')

    drop_out = Dropout(0.3, name='dropout')(e4)

    # pass the input through a BiLSTM model through the previous layer
    bilstm = Bidirectional(LSTM(lstm_units, return_sequences=True))(drop_out)

    # add another drop-out layer
    drop_out = Dropout(0.1)(bilstm)

    # here we add a custom layer which connects the last input of the first table as the input of the second table
    t2 = Lambda(get_H_n, output_shape=(k,), name='h_n')(drop_out)

    # add the layer which encodes the output labels
    Y = Lambda(get_Y, arguments={'xmaxlen': col_length}, name='Y', output_shape=(col_length, k))(drop_out)
    # add the layer which encodes the weight parameter from the LSTMs cells and the hidden states
    Whn = Dense(k, W_regularizer=l2(0.01), name="Wh_n")(t2)
    Whn_x_e = RepeatVector(col_length, name="Wh_n_x_e")(Whn)

    # add the attention layer
    WY = TimeDistributed(Dense(2 * lstm_units, W_regularizer=l2(0.01)), name="WY")(Y)
    merged = merge([Whn_x_e, WY], name='merged', mode='sum')
    M = Activation('tanh', name='M')(merged)

    alpha_ = TimeDistributed(Dense(1, activation='linear'), name="alpha_")(M)
    flat_alpha = Flatten(name="flat_alpha")(alpha_)
    alpha = Dense(col_length, activation='softmax', name="alpha")(flat_alpha)

    Y_trans = Permute((2, 1), name="y_trans")(Y)  # of shape (None,300,20)
    r_ = merge([Y_trans, alpha], output_shape=(k, 1), name="r_", mode=get_R)
    r = Reshape((k,), name="r")(r_)

    Wr = Dense(k, W_regularizer=l2(0.01))(r)
    Wh = Dense(k, W_regularizer=l2(0.01))(t2)
    merged = merge([Wr, Wh], mode='sum')
    h_star = Activation('tanh')(merged)

    out = Dense(3, activation='softmax')(h_star)
    output = out

    model = Model(input=[main_input, subject_input, value_input], output=output)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['acc'])
    return model

