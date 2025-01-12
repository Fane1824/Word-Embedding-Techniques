import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
import pickle

def load_data(file_path, num_pos, num_neg):
    data = pd.read_csv(file_path)
    # Filter positive and negative reviews
    pos_reviews = data[data['polarity'] == 1]['review'][:num_pos]
    neg_reviews = data[data['polarity'] == 0]['review'][:num_neg]
    texts = pd.concat([pos_reviews, neg_reviews]).tolist()
    labels = [1]*len(pos_reviews) + [0]*len(neg_reviews)
    return texts, labels

# Load training data
train_texts, train_labels = load_data('../allocine/train.csv', 12500, 12500)

# Tokenize the texts
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_texts)
word_index = tokenizer.word_index

# Convert texts to sequences
X_train_sequences = tokenizer.texts_to_sequences(train_texts)

# Pad sequences
maxlen = 100
X_train_padded = pad_sequences(X_train_sequences, maxlen=maxlen)

# Prepare the embedding matrix using GloVe
embedding_dim = 100
embedding_index = {}

# Load GloVe embeddings
glove_file = '../../../glove.6B.100d.txt'
with open(glove_file, 'r', encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coeffs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coeffs

# Create embedding matrix
vocab_size = len(word_index) + 1
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
    else:
        embedding_matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim, ))

# Build BiLSTM model
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                    weights=[embedding_matrix],
                    input_length=maxlen, trainable=False))
model.add(Bidirectional(LSTM(units=64, dropout=0.5, recurrent_dropout=0.5)))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_padded, np.array(train_labels),
          epochs=5, batch_size=64, validation_split=0.1)

# Save the model
model.save('sentiment_model.h5')

# Save the tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)