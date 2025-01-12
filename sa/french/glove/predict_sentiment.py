import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import pickle

def load_data(file_path, num_pos, num_neg):
    data = pd.read_csv(file_path)
    # Filter positive and negative reviews
    pos_reviews = data[data['polarity'] == 1]['review'][:num_pos]
    neg_reviews = data[data['polarity'] == 0]['review'][:num_neg]
    texts = pd.concat([pos_reviews, neg_reviews]).tolist()
    labels = [1]*len(pos_reviews) + [0]*len(neg_reviews)
    return texts, labels

# Load test data
test_texts, test_labels = load_data('../allocine/test.csv', 3125, 3125)

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
pred_labels = (predictions > 0.5).astype(int).flatten()

# Generate confusion matrix and classification report
cm = confusion_matrix(test_labels, pred_labels)
print('Confusion Matrix')
print(cm)

print('\nClassification Report')
print(classification_report(test_labels, pred_labels, target_names=['Negative', 'Positive']))