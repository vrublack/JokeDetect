'''This script loads pre-trained word embeddings (GloVe embeddings)
into a frozen Keras Embedding layer, and uses it to
train a text classification model on the 20 Newsgroup dataset
(classication of newsgroup messages into 20 different categories).

GloVe embedding data can be found at:
http://nlp.stanford.edu/data/glove.6B.zip
(source page: http://nlp.stanford.edu/projects/glove/)

20 Newsgroup data can be found at:
http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html
'''

from __future__ import print_function
import os
import random

import numpy as np

np.random.seed(1337)
random.seed(1337)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
import sys
import pickle

BASE_DIR = ''
GLOVE_DIR = BASE_DIR + 'glove.6B/'
JOKE_DIR = BASE_DIR + 'joke-data/'
JOKE_FNAMES = ['humorous_jokes.pickle', 'short_oneliners.pickle']
NONJOKE_FNAMES = ['short_wiki_sentences.pickle', 'movie_dialogs.txt']
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

# first, build index mapping words in the embeddings set
# to their embedding vector

print('Indexing word vectors.')

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    # TODO remove
    # if len(embeddings_index) > 100:
    #     break
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# second, prepare text samples and their labels
print('Processing text dataset')

texts = []  # list of text samples
labels = []  # list of label ids

for name in JOKE_FNAMES:
    path = os.path.join(JOKE_DIR, name)
    with open(path, 'rb') as f:
        b = pickle.load(f, encoding='latin1')
        for l in b:
            texts.append(l)
            labels.append(1)

for name in NONJOKE_FNAMES:
    path = os.path.join(JOKE_DIR, name)
    mode = 'r' if name.endswith('.txt') else 'rb'
    with open(path, mode) as f:
        if name.endswith('.txt'):
            b = []
            l = f.readline()
            while l:
                b.append(l)
                try:
                    l = f.readline()
                except UnicodeDecodeError:
                    l = '???'
        else:
            b = pickle.load(f, encoding='latin1')
        if len(b) > 30000:
            b = b[:30000]
        for l in b:
            texts.append(l)
            labels.append(0)

# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
# np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[:]
y_train = labels[:]

# prepare embedding matrix
nb_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(nb_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(35)(x)
x = Flatten()(x)
x = Dense(50, activation='relu')(x)
preds = Dense(2, activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# happy learning!
model.load_weights('weights_words')

def make_predict_data():
    texts = []  # list of text samples
    labels = []  # list of label ids

    texts.append("What's the difference between a cow and a truck? The wheels.")
    labels.append(1)
    texts.append("Knock knock. Who's there? A watch!")
    labels.append(1)
    texts.append("I am the weakest link. Too bad this is not true for you.")
    labels.append(1)
    texts.append("Why did the chicken cross the road?")
    labels.append(1)
    texts.append("Why did the chicken cross the road?")
    labels.append(1)
    texts.append("Light travels faster than sound. This is why some people appear bright until you hear them speak.")
    labels.append(1)
    texts.append("Men have two emotions: Hungry and Horny. If you see him without an erection, make him a sandwich.")
    labels.append(1)
    texts.append("In the sixteenths century, Poland has seen a big influx in immigrants.")
    labels.append(0)
    texts.append("The united kingdom has a population of about 300 people")
    labels.append(0)
    texts.append("If you want to exit you can just go there.")
    labels.append(0)
    texts.append("Go to the homepage and download the song. It's an mp3.")
    labels.append(0)
    texts.append("We've had a quick scout around the internet for the best one-liners we could find and these were the ones that made us chortle")
    labels.append(0)
    texts.append("This looks absolutely heavenly. Sometimes simpler is better.")
    labels.append(0)


    # finally, vectorize the text samples into a 2D integer tensor
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    labels = to_categorical(np.asarray(labels))

    # split the data into a training set and a validation set
    indices = np.arange(data.shape[0])
    data = data[indices]
    labels = labels[indices]

    x_train = data[:]
    y_train = labels[:]

    return x_train, y_train

del embeddings_index

x_train, y_train = make_predict_data()

predict = model.predict(x_train)

print('Predictions: \n' + str(predict))
