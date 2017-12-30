# -*- coding:utf-8 -*-
########################################
## import packages
########################################
import os
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout
from keras.layers.merge import concatenate
from keras.models import Model, load_model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.preprocessing.text
import pickle


########################################
# set directories and parameters
########################################
# DATA_DIR = '../dataset/'
EMBEDDING_FILE = '/media/jlan/E/Projects/nlp/sentence_similarity/dataset/w2v_model.bin'
TRAIN_DATA_FILE = '/media/jlan/E/Projects/nlp/sentence_similarity/dataset/dataset.csv'
TEST_DATA_FILE = '/media/jlan/E/Projects/nlp/sentence_similarity/dataset/test.csv'
MAX_SEQUENCE_LENGTH = 30
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.1


num_lstm = 175
num_dense = 100
rate_drop_lstm = 0.15
rate_drop_dense = 0.15

act = 'relu'
re_weight = True  # whether to re-weight classes to fit the 17.5% share in test set

# STAMP = '../model/lstm/lstm_%d_%d_%.2f_%.2f' % (num_lstm, num_dense, rate_drop_lstm,
#                                                 rate_drop_dense)
STAMP = './model/bow'
save = True
load_tokenizer = False
save_path = "./model"
tokenizer_name = "tokenizer.pkl"
embedding_matrix_path = "./model/embedding_matrix.npy"


def data_prepare(datafile):
    """
    从文件中读取数据
    :param datafile:
    :return:
    """
    data = pd.read_csv(datafile)
    sentences1 = data['sentence1'].values
    sentences2 = data['sentence2'].values
    labels = data['similarity'].values
    return sentences1, sentences2, labels


def tokenize(sentences):
    """
    获取所有文本中的词语
    :param sentences:
    :return:
    """
    if load_tokenizer:
        print('Load tokenizer...')
        tokenizer = pickle.load(open(os.path.join(save_path, tokenizer_name), 'rb'))
    else:
        print("Fit tokenizer...")
        tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=False)
        tokenizer.fit_on_texts(sentences)
        if save:
            print("Save tokenizer...")
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            pickle.dump(tokenizer, open(os.path.join(save_path, tokenizer_name), "wb"))
    print('Found %s unique tokens' % len(tokenizer.word_index))
    return tokenizer


def sent2seq(tokenizer, sentences):
    """
    把句子转换成序列，如‘如何 来 防治 水稻 稻瘟病’----->[6, 383, 2, 1, 12]
    :param tokenizer:
    :return:
    """
    sequences = tokenizer.texts_to_sequences(sentences)
    sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH) # 维度统一为MAX_SEQUENCE_LENGTH，不足的补0
    return sequences


def w2v(tokenizer, nb_words):
    """
    prepare embeddings
    :param tokenizer:
    :param nb_words:
    :return:
    """
    print('Preparing embedding matrix')
    word2vec = Word2Vec.load(EMBEDDING_FILE)
    embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
    for word, i in tokenizer.word_index.items():
        if word in word2vec.wv.vocab:
            embedding_matrix[i] = word2vec.wv.word_vec(word)
        else:
            print(word)
    print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

    np.save(embedding_matrix_path, embedding_matrix)
    return embedding_matrix


def get_model(nb_words, embedding_matrix):
    """
    定义模型结构
    :param nb_words:
    :param embedding_matrix:
    :return:
    """
    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                # weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)

    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    x1 = lstm_layer(embedded_sequences_1)

    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)
    y1 = lstm_layer(embedded_sequences_2)

    merged = concatenate([x1, y1])
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)

    merged = Dense(num_dense, activation=act)(merged)
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)
    preds = Dense(1, activation='sigmoid')(merged)

    model = Model(inputs=[sequence_1_input, sequence_2_input],
                  outputs=preds)
    model.compile(loss='binary_crossentropy',
                  optimizer='nadam',
                  metrics=['acc'])
    model.summary()
    return model


def train_model(model, seq1, seq2, labels):
    """
    训练模型
    :param model:
    :param seq1:
    :param seq2:
    :param labels:
    :param test_seq1:
    :param test_seq2:
    :param test_labels:
    :return:
    """
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    bst_model_path = STAMP + '.h5'
    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=False)

    hist = model.fit([seq1, seq2], labels,
                     # validation_data=([test_seq1[:-100], test_seq2[:-100]], test_labels[:-100]),
                     validation_split=0.2,
                     epochs=50, batch_size=10, shuffle=True, callbacks=[early_stopping, model_checkpoint])

    model.load_weights(bst_model_path)
    bst_score = min(hist.history['loss'])
    bst_acc = max(hist.history['acc'])
    print(bst_acc, bst_score)
    print("Test score", min(hist.history["val_loss"]))
    print("Test acc", max(hist.history["val_acc"]))


def test(model, test_sentences1, test_sentences2, test_seq1, test_seq2, test_labels):
    predicts = model.predict([test_seq1, test_seq2], batch_size=10, verbose=1)
    pres = np.array([1 if p[0]>0.5 else 0 for p in predicts])
    for i in range(len(test_labels[-100:])):
        print("t1: {}, t2: {}, score: {}, real_sim: {}".
              format(test_sentences1[i], test_sentences2[i], predicts[i], test_labels[i])
              )

    print(sum(pres==test_labels))


def evaluate(model, test_seq1, test_seq2, test_labels):
    score = model.evaluate([test_seq1, test_seq2], test_labels,batch_size=10)
    print(score)


def main():
    print('\n从文件中读取数据..............................')
    sentences1, sentences2, labels = data_prepare(TRAIN_DATA_FILE)
    # test_sentences1, test_sentences2, test_labels = data_prepare(TEST_DATA_FILE)
    print('Found %s texts in train.csv' % len(sentences1))

    sentence_all = np.concatenate((sentences1, sentences2), axis=0)

    print('\n获取所有文本中的词语..........................')
    tokenizer = tokenize(sentence_all)

    nb_words = min(MAX_NB_WORDS, len(tokenizer.word_index)) + 1

    print('\n把句子转换成序列， 并进行长度补全...............')
    seq1 = sent2seq(tokenizer, sentences1)
    seq2 = sent2seq(tokenizer, sentences2)

    print('\n计算每个词语的向量............................')
    # embedding_matrix = w2v(tokenizer, nb_words)
    # # embedding_matrix = np.ones((nb_words, EMBEDDING_DIM))
    #
    # print('\n设计模型结构..................................')
    # model = get_model(nb_words, embedding_matrix)
    #
    # print('\n训练模型.....................................')
    # train_model(model, seq1[:-100], seq2[:-100], labels[:-100])

    print('\n测试模型.....................................')
    model = load_model('./model/bow.h5')
    test(model, sentences1[-1000:], sentences2[-1000:], seq1[-1000:], seq2[-1000:], labels[-1000:])
    evaluate(model, seq1[-2000:], seq2[-2000:], labels[-2000:])

if __name__ == '__main__':
    main()
