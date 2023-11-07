# a. Data preparation

import numpy as np
import keras.backend as K #imp
from keras.models import Sequential
from keras.layers import Dense, Embedding, Lambda
import tensorflow as tf
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
import gensim

data = open(r"../corona.txt")
data

corona_data = [text for text in data if text.count(' ') >= 2]
print(corona_data)

vectorize = Tokenizer()
vectorize.fit_on_texts(corona_data)
corona_data = vectorize.texts_to_sequences(corona_data)

total_vocab = sum(len(s) for s in corona_data)
word_count = len(vectorize.index_word)+1
print(corona_data)

print(total_vocab)
print(word_count)

window_size = 2

# b. Generate training data

# Defining utility to generate context word pairs
def cbow_model(data, window_size, total_vocab):
    total_length = window_size*2
    for text in data:
        text_len = len(text)
#         print("zero",text)
        for idx, word in enumerate(text):
#             print("first",idx,word)
            context_word = []
            target   = []
            begin = idx - window_size
            end = idx + window_size + 1
            context_word.append([
                text[i]
                for i in range(begin, end)
                if 0 <= i < text_len
                and i != idx
            ])
            target.append(word)
#             print("second",context_word,target)
            contextual = pad_sequences(
                context_word,
                maxlen=total_length
            )
            final_target = tf.keras.utils.to_categorical(
                target,
                total_vocab
            )
#             print("third",contextual,final_target)
            yield(contextual, final_target)

# c. train model

# Defining the model architecture
model = Sequential()
model.add(
    Embedding(
        input_dim=total_vocab,
        output_dim=100,
        input_length=window_size*2
    )
)
model.add(
    Lambda(
        lambda x: K.mean(x, axis=1),
        output_shape=(100,)
    )
)
model.add(
    Dense(
        total_vocab,
        activation='softmax'
    )
)

model.summary()

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam'
)

for i in range(10):
    cost = 0
    for x, y in cbow_model(corona_data, window_size, total_vocab):
        cost += model.train_on_batch(x, y)
    print("Epoch ", i,"\t: ", cost)

dimensions = 100
vect_file = open('../vectors.txt', 'w')
print(vect_file.write('{} {}\n'.format(102, dimensions)))

weights = model.get_weights()[0]
for text, i in vectorize.word_index.items():
    final_vec = ' '.join(map(str, list(weights[i, :])))
    vect_file.write('{} {}\n'.format(text, final_vec))
vect_file.close()

# d. Output

cbow_output = gensim.models.KeyedVectors.load_word2vec_format(
    'vectors.txt',
    binary=False
)

print(cbow_output.most_similar(positive=['speed']))