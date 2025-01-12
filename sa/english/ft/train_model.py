import os
import numpy as np
from gensim.models import FastText
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
import pickle

def load_data(directory):
    texts = []
    labels = []
    for label_type in ['neg', 'pos']:
        dir_name = os.path.join(directory, label_type)
        for fname in os.listdir(dir_name):
            if fname.endswith('.txt'):
                with open(os.path.join(dir_name, fname), encoding='utf8') as f:
                    texts.append(f.read())
                labels.append(0 if label_type == 'neg' else 1)
    return texts, labels

# Load training data
train_texts, train_labels = load_data('../aclImdb/train')

# Tokenize the texts
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_texts)
word_index = tokenizer.word_index

# Convert texts to sequences
X_train_sequences = tokenizer.texts_to_sequences(train_texts)

# Pad sequences
maxlen = 100
X_train_padded = pad_sequences(X_train_sequences, maxlen=maxlen)

# Train FastText model
train_tokens = [text.split() for text in train_texts]
embedding_dim = 100
fasttext_model = FastText(sentences=train_tokens, vector_size=embedding_dim, window=5, min_count=1, workers=4)

# Create embedding matrix
vocab_size = len(word_index) + 1
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in word_index.items():
    if word in fasttext_model.wv:
        embedding_matrix[i] = fasttext_model.wv[word]
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

# Save the FastText model
fasttext_model.save('fasttext.model')