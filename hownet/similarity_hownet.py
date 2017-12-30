import numpy as np
import pandas as pd
from hownet import similar_sentence

TRAIN_DATA_FILE = '/media/jlan/E/Projects/nlp/sentence_similarity/dataset/small.csv'


def data_prepare(datafile):
    data = pd.read_csv(datafile)
    sentences1 = data['sentence1'].values
    sentences2 = data['sentence2'].values
    labels = data['similarity'].values
    return sentences1, sentences2, labels


def main():
    predicts = []
    sentences1, sentences2, labels = data_prepare(TRAIN_DATA_FILE)
    for sent1, sent2 in zip(sentences1, sentences2):
        score = similar_sentence(sent1.split(), sent2.split())
        print('{}, {}, {}'.format(sent1, sent2, score))
        predicts.append(score)
    predicts = [1 if pre>0.5 else 0 for pre in predicts]
    predicts = np.array(predicts)
    print(sum(predicts==labels))


if __name__ == '__main__':
    main()