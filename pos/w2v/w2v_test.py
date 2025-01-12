import numpy as np
import pandas as pd
from gensim.models import Word2Vec
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import time
import os

class POSDataset(Dataset):
    def __init__(self, sentences, tags, word2idx, tag2idx):
        self.sentences = sentences
        self.tags = tags
        self.word2idx = word2idx
        self.tag2idx = tag2idx
        
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        words = self.sentences[idx]
        tags = self.tags[idx]
        x = torch.tensor([self.word2idx.get(w, self.word2idx['<UNK>']) for w in words])
        y = torch.tensor([self.tag2idx.get(t, self.tag2idx['<UNK_TAG>']) for t in tags])
        return x, y

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, tag_size, embedding_dim, hidden_dim, pretrained_embeddings):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, 
                           bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tag_size)
        
    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        tag_space = self.hidden2tag(lstm_out)
        return tag_space

def load_data(filename):
    words, tags = [], []
    with open(filename, 'r') as f:
        for line in f:
            if line.strip():
                word, tag = line.strip().split('\t')
                words.append(word)
                tags.append(tag)
    return words, tags

def main():
    language = input("Which language do you want (English, French): ").lower()

    start_time = time.time()
    
    if language == 'french':
        train_file = '../train_fr.txt'
        test_file = '../test_fr.txt'
    else:
        train_file = '../train_en.txt'
        test_file = '../test_en.txt'
    
    # Load training data
    train_words, train_tags = load_data(train_file)
    # Load testing data
    test_words, test_tags = load_data(test_file)
    
    # Create sentences
    train_sentences = [[word] for word in train_words]
    train_tag_sequences = [[tag] for tag in train_tags]
    
    test_sentences = [[word] for word in test_words]
    test_tag_sequences = [[tag] for tag in test_tags]
    
    # Train Word2Vec
    word2vec = Word2Vec(train_sentences, vector_size=100, window=5, min_count=1)
    
    # Create vocabularies
    unique_words = set(train_words)
    word2idx = {word: idx+1 for idx, word in enumerate(unique_words)}  # 0 reserved for <UNK>
    word2idx['<UNK>'] = 0
    
    unique_tags = set(train_tags).union(set(test_tags))
    tag2idx = {tag: idx for idx, tag in enumerate(unique_tags)}
    tag2idx['<UNK_TAG>'] = len(tag2idx)
    idx2tag = {idx: tag for tag, idx in tag2idx.items()}
    
    # Prepare embedding matrix
    embedding_dim = 100
    embedding_matrix = np.zeros((len(word2idx), embedding_dim))
    for word, idx in word2idx.items():
        if word in word2vec.wv:
            embedding_matrix[idx] = word2vec.wv[word]
        else:
            embedding_matrix[idx] = np.random.uniform(-0.25, 0.25, embedding_dim)
    
    # Create datasets
    train_dataset = POSDataset(train_sentences, train_tag_sequences, word2idx, tag2idx)
    test_dataset = POSDataset(test_sentences, test_tag_sequences, word2idx, tag2idx)
    
    # Initialize model
    model = BiLSTM(
        vocab_size=len(word2idx),
        tag_size=len(tag2idx),
        embedding_dim=100,
        hidden_dim=200,
        pretrained_embeddings=embedding_matrix
    )
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=tag2idx['<UNK_TAG>'])
    optimizer = torch.optim.Adam(model.parameters())
    
    # Training loop
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    for epoch in range(10):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs.view(-1, len(tag2idx)), batch_y.view(-1))
            loss.backward()
            optimizer.step()
    
    # Evaluation
    model.eval()
    all_predictions = []
    all_true_tags = []
    
    test_loader = DataLoader(test_dataset, batch_size=32)
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            predictions = torch.argmax(outputs, dim=-1)
            all_predictions.extend(predictions.tolist())
            all_true_tags.extend(batch_y.tolist())
    
    # Convert predictions to tags
    pred_tags = [[idx2tag.get(idx, '<UNK_TAG>') for idx in seq] for seq in all_predictions]
    true_tags = [[idx2tag.get(idx, '<UNK_TAG>') for idx in seq] for seq in all_true_tags]
    
    # Flatten for metrics
    pred_tags_flat = [tag for seq in pred_tags for tag in seq]
    true_tags_flat = [tag for seq in true_tags for tag in seq]
    
    # Define the list of all possible labels excluding '<UNK_TAG>'
    labels = sorted([tag for tag in tag2idx.keys() if tag != '<UNK_TAG>'])
    
    # Print metrics
    print("\nClassification Report:")
    print(classification_report(true_tags_flat, pred_tags_flat, labels=labels, zero_division=0))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(true_tags_flat, pred_tags_flat, labels=labels)
    cm_df = pd.DataFrame(
        cm,
        index=labels,
        columns=labels
    )
    print(cm_df)
    end_time = time.time()
    print("Time taken: ", end_time - start_time)

if __name__ == "__main__":
    main()