import numpy as np
from datasets import load_dataset
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# Load the saved model and artifacts
model = load_model('ner_model.h5')

with open('word_index.pkl', 'rb') as f:
    word_index = pickle.load(f)

with open('label_mappings.pkl', 'rb') as f:
    mappings = pickle.load(f)
label2idx = mappings['label2idx']
idx2label = mappings['idx2label']

label_list = [idx2label[i] for i in range(len(idx2label))]

# Load the test data
dataset = load_dataset('conll2003')

def get_sentences_and_labels(split):
    sentences = []
    labels = []
    for item in dataset[split]:
        sentences.append(item['tokens'])
        labels.append(item['ner_tags'])
    return sentences, labels

test_sentences, test_labels = get_sentences_and_labels('test')

# Encode sentences and labels
max_len = 75

def encode_sentences(sentences):
    sequences = []
    for sentence in sentences:
        seq = [word_index.get(word, 0) for word in sentence]
        sequences.append(seq)
    return pad_sequences(sequences, maxlen=max_len, padding='post')

def encode_labels(labels):
    sequences = []
    for label_seq in labels:
        sequences.append([label for label in label_seq])
    return pad_sequences(sequences, maxlen=max_len, padding='post', value=label2idx['O'])

X_test = encode_sentences(test_sentences)
y_test = encode_labels(test_labels)
y_test_flat = y_test.flatten()

# Predict and evaluate
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=-1)
y_pred_flat = y_pred_labels.flatten()


cm = confusion_matrix(y_test_flat, y_pred_flat)
cm_df = pd.DataFrame(cm, index=label_list, columns=label_list)
print('Confusion Matrix:')
print(cm_df)
print('\nClassification Report')
print(classification_report(y_test_flat, y_pred_flat, target_names=label_list, zero_division=0))