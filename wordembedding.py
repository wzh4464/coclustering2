import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np

class SkipGramDataset(Dataset):
    def __init__(self, sentences, window_size=2):
        self.window_size = window_size
        self.word_counts = Counter([word for sentence in sentences for word in sentence])
        self.vocab = list(self.word_counts.keys())
        self.word_to_index = {word: i for i, word in enumerate(self.vocab)}
        self.index_to_word = {i: word for word, i in self.word_to_index.items()}
        self.data = self.generate_data(sentences, window_size)

    def generate_data(self, sentences, window_size):
        data = []
        for sentence in sentences:
            indices = [self.word_to_index[word] for word in sentence]
            for i in range(len(indices)):
                for j in range(max(0, i - window_size), min(i + window_size + 1, len(indices))):
                    if i != j:
                        data.append((indices[i], indices[j]))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        x = self.embeddings(x)
        x = self.linear(x)
        return x


def train(model, data_loader, epochs, device):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(epochs):
        total_loss = 0.0
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(data_loader)}")

# read sentences from processed/output.txt
# each line is a sentence

with open('processed/output.txt', 'r') as f:
    sentences = [line.split() for line in f.readlines()]

dataset = SkipGramDataset(sentences)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = SkipGramModel(len(dataset.vocab), embedding_dim=50)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train(model, data_loader, epochs=50, device=device)

# get embedding representation of each sentence and save to a matrix. Also save to file line by line

Matrix = np.zeros((len(sentences), 50))
with open('embeddings.txt', 'w') as f:
    for i, sentence in enumerate(sentences):
        sentence = torch.tensor([dataset.word_to_index[word] for word in sentence])
        sentence = sentence.to(device)
        embedding = model.embeddings(sentence).mean(dim=0)
        Matrix[i] = embedding.detach().cpu().numpy()
        f.write(' '.join([str(x) for x in embedding.detach().cpu().numpy()]))
        f.write('\n')
    
