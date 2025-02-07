import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import re





# Load text file
with open('train.txt', 'r', encoding='utf-8') as f:
    text = f.read().lower()

# Tokenize the text into words
def tokenize(text):
    return re.findall(r'\w+', text)

tokens = tokenize(text)
 
# Build vocabulary with an <UNK> token for unknown words
vocab = ['<UNK>'] + sorted(set(tokens))
vocab_size = len(vocab)
word_to_idx = {word: idx for idx, word in enumerate(vocab)}
idx_to_word = {idx: word for idx, word in enumerate(vocab)}

# Convert tokens to indices
indices = [word_to_idx.get(word, word_to_idx['<UNK>']) for word in tokens]


sequence_length = 3  # Adjust as needed
sequences = []
targets = []

for i in range(len(indices) - sequence_length):
    seq = indices[i:i+sequence_length]
    target = indices[i+sequence_length]
    sequences.append(seq)
    targets.append(target)

class TextDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.long)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

# Create DataLoader
dataset = TextDataset(sequences, targets)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


class NextWordPredictor(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Get the last time step's output
        out = self.linear(out)
        return out

# Initialize the model
model = NextWordPredictor(vocab_size, embed_dim=128, hidden_size=256)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)



num_epochs = 100

for epoch in range(num_epochs):
    total_loss = 0
    for batch_inputs, batch_targets in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')




def predict_next_word(model, seed_text, word_to_idx, idx_to_word, sequence_length=3):
    model.eval()
    tokens = tokenize(seed_text.lower())
    tokens = tokens[-sequence_length:]  # Truncate to the last 'sequence_length' tokens
    
    # Convert tokens to indices, handling unknown words
    indices = [word_to_idx.get(word, word_to_idx['<UNK>']) for word in tokens]
    
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
    predicted_idx = output.argmax(1).item()
    return idx_to_word[predicted_idx]

# Example usage
seed_text = input("Enter your seed text: ")
predicted_word = predict_next_word(model, seed_text, word_to_idx, idx_to_word)
print(f"Seed: '{seed_text}' -> Predicted next word: '{predicted_word}'")
