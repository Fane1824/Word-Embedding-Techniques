import numpy as np
from datasets import load_dataset
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, TimeDistributed, Dense
import pickle

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

# Create label mapping
label_list = dataset['train'].features['ner_tags'].feature.names
label2idx = {label: idx for idx, label in enumerate(label_list)}
idx2label = {idx: label for label, idx in label2idx.items()}

# Tokenize the sentences
tokenizer = Tokenizer(lower=False, oov_token=None)
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index

# Load GloVe embeddings
embedding_dim = 100
embedding_index = {}

glove_file = '../../../glove.6B.100d.txt'
with open(glove_file, 'r', encoding='utf8') as f:
    for line in f:
        values = line.strip().split()
        word = values[0]
        coeffs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coeffs

# Prepare embedding matrix
vocab_size = len(word_index) + 1
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, idx in word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[idx] = embedding_vector
    else:
        embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim, ))

# Encode sentences and labels
max_len = 75

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