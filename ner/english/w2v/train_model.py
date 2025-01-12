import numpy as np
import random
import tensorflow as tf
from datasets import load_dataset
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, TimeDistributed, Dense
import pickle

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Define maximum sequence length
max_len = 75

# Load the CoNLL-2003 dataset
dataset = load_dataset('conll2003')

# Prepare the data
def get_sentences_and_labels(split):
    sentences = []
    labels = []
    for item in dataset[split]:
        sentences.append(item['tokens'])
        labels.append(item['ner_tags'])
    return sentences, labels

train_sentences, train_labels = get_sentences_and_labels('train')

# Create label mappings
label_list = dataset['train'].features['ner_tags'].feature.names
label2idx = {label: idx for idx, label in enumerate(label_list)}
idx2label = {idx: label for label, idx in label2idx.items()}

# Tokenize the sentences
tokenizer = Tokenizer(lower=False, oov_token=None)
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index

# Train Word2Vec embeddings
embedding_dim = 100
train_tokens = train_sentences  # Sentences are already tokenized
w2v_model = Word2Vec(sentences=train_tokens, vector_size=embedding_dim, window=5, min_count=1, workers=4, seed=42)

# Create embedding matrix
vocab_size = len(word_index) + 1  # Plus one for padding token
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, idx in word_index.items():
    if word in w2v_model.wv:
        embedding_matrix[idx] = w2v_model.wv[word]
    else:
        # Random initialization for words not in Word2Vec model
        embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim, ))

# Encode sentences and labels
def encode_sentences(sentences):
    sequences = tokenizer.texts_to_sequences(sentences)
    return pad_sequences(sequences, maxlen=max_len, padding='post')

def encode_labels(labels):
    sequences = []
    for label_seq in labels:
        sequences.append([label for label in label_seq])
    return pad_sequences(sequences, maxlen=max_len, padding='post', value=label2idx['O'])

X_train = encode_sentences(train_sentences)
y_train = encode_labels(train_labels)
y_train = to_categorical(y_train, num_classes=len(label_list))

# Build the BiLSTM model
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                    weights=[embedding_matrix], input_length=max_len, trainable=False))
model.add(Bidirectional(LSTM(units=64, return_sequences=True, recurrent_dropout=0.1)))
model.add(TimeDistributed(Dense(len(label_list), activation='softmax')))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=5, validation_split=0.1)

# Save the model and artifacts
model.save('ner_model.h5')

# Save the tokenizer
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# Save label mappings
with open('label_mappings.pkl', 'wb') as f:
    pickle.dump({'label2idx': label2idx, 'idx2label': idx2label}, f)

# Save word index
with open('word_index.pkl', 'wb') as f:
    pickle.dump(word_index, f)

# Save the Word2Vec model
w2v_model.save('word2vec.model')