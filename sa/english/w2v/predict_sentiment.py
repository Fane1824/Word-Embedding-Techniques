import os
import numpy as np
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
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

# Load test data
test_texts, test_labels = load_data('../aclImdb/test')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Convert texts to sequences
X_test_sequences = tokenizer.texts_to_sequences(test_texts)

# Pad sequences
maxlen = 100  # Same as in training
X_test_padded = pad_sequences(X_test_sequences, maxlen=maxlen)

# Load the model
model = load_model('sentiment_model.h5')

# Predict on test data
predictions = model.predict(X_test_padded)
pred_labels = (predictions > 0.5).astype(int)

# Generate confusion matrix and classification report
cm = confusion_matrix(test_labels, pred_labels)
print('Confusion Matrix')
print(cm)

print('\nClassification Report')
print(classification_report(test_labels, pred_labels, target_names=['Negative', 'Positive']))