import numpy as np
from datasets import load_dataset
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd  # Added import for pandas

# Load the saved model and artifacts
model = load_model('ner_model.h5')

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

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
    # Join tokens back into sentences for tokenizer
    joined_sentences = [' '.join(sentence) for sentence in sentences]
    sequences = tokenizer.texts_to_sequences(joined_sentences)
    return pad_sequences(sequences, maxlen=max_len, padding='post')

def encode_labels(labels):
    sequences = []
    for label_seq in labels:
        sequences.append([label for label in label_seq])
    return pad_sequences(sequences, maxlen=max_len, padding='post', value=label2idx['O'])

X_test = encode_sentences(test_sentences)
y_test = encode_labels(test_labels)
y_test_flat = y_test.flatten()
y_test_flat = y_test_flat[y_test_flat != -1]  # Exclude padding if applicable

# Predict and evaluate
y_pred = model.predict(X_test, verbose=1)
y_pred_labels = np.argmax(y_pred, axis=-1)
y_pred_flat = y_pred_labels.flatten()
y_pred_flat = y_pred_flat[y_test.flatten() != -1]  # Exclude padding if applicable

# Ensure that y_test_flat and y_pred_flat have the same length
min_length = min(len(y_test_flat), len(y_pred_flat))
y_test_flat = y_test_flat[:min_length]
y_pred_flat = y_pred_flat[:min_length]

# Map indices to label names
y_true_labels = [idx2label[idx] for idx in y_test_flat]
y_pred_labels = [idx2label[idx] for idx in y_pred_flat]

cm = confusion_matrix(y_test_flat, y_pred_flat)
cm_df = pd.DataFrame(cm, index=label_list, columns=label_list)
print('Confusion Matrix:')
print(cm_df)
print('\nClassification Report')
print(classification_report(y_test_flat, y_pred_flat, target_names=label_list, zero_division=0))