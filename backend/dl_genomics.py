import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os

class GenomicCNN(nn.Module):
    def __init__(self, seq_len=1000, num_classes=1):
        super(GenomicCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=32, kernel_size=8, stride=1, padding=0)
        self.pool1 = nn.MaxPool1d(kernel_size=4)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=8, stride=1, padding=0)
        self.pool2 = nn.MaxPool1d(kernel_size=4)
        
        def conv1d_out(size, kernel_size=8, stride=1, padding=0):
            return (size + 2*padding - kernel_size) // stride + 1
        
        def pool1d_out(size, kernel_size=4, stride=4, padding=0):
            return (size + 2*padding - kernel_size) // stride + 1
            
        c1 = conv1d_out(seq_len)
        p1 = pool1d_out(c1, stride=4)
        c2 = conv1d_out(p1)
        p2 = pool1d_out(c2, stride=4)
        
        self.flattened_dim = 64 * p2
        
        self.fc1 = nn.Linear(self.flattened_dim, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, self.flattened_dim)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return self.sigmoid(x)

def sequence_to_tensor(sequence, max_len=1000):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    sequence = sequence.upper().replace('\n', '').replace('>', '')
    if sequence.startswith('ISOLATE'):
        sequence = sequence[sequence.find('\n')+1:]
    sequence = "".join([c for c in sequence if c in mapping])
    if len(sequence) > max_len:
        sequence = sequence[:max_len]
    else:
        sequence = sequence.ljust(max_len, 'A')
        
    tensor = torch.zeros(4, max_len)
    for i, char in enumerate(sequence):
        idx = mapping.get(char, 0)
        tensor[idx, i] = 1.0
    return tensor

def train_synthetic(dynamic_motifs=None):
    print("Initializing PyTorch 1D-CNN Genomic Extractor...")
    seq_len = 1000
    model = GenomicCNN(seq_len=seq_len)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    if dynamic_motifs is None or len(dynamic_motifs) == 0:
        print("No dynamic motifs provided. Using fallbacks.")
        dynamic_motifs = [
            "ATGGAATTGCCCAATATTATGCACCCGGTCGCGAAGCTGAGC", # NDM-1
            "GGCGTATTCGACCTGTAT" # parC
        ]
        
    print(f"Injecting {len(dynamic_motifs)} explicit clinical genomic fragments into synthetic FASTA arrays...")
    
    num_samples = 3000
    X = []
    y = []
    
    for i in range(num_samples):
        seq = "".join(np.random.choice(['A', 'C', 'G', 'T'], size=seq_len))
        is_resistant = np.random.rand() > 0.5
        
        if is_resistant:
            motif = np.random.choice(dynamic_motifs)
            if len(motif) < seq_len:
                insert_pos = np.random.randint(0, seq_len - len(motif))
                seq = seq[:insert_pos] + motif + seq[insert_pos + len(motif):]
        
        X.append(sequence_to_tensor(seq, seq_len))
        y.append(1.0 if is_resistant else 0.0)
        
    X_train = torch.stack(X)
    y_train = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    
    print("Training 1D-CNN on sequences...")
    model.train()
    epochs = 4
    batch_size = 64
    for epoch in range(epochs):
        permutation = torch.randperm(X_train.size()[0])
        epoch_loss = 0
        for i in range(0, X_train.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train[indices], y_train[indices]
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(X_train):.4f}")
        
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "models"), exist_ok=True)
    save_path = os.path.join(os.path.dirname(__file__), "..", "models", "genomic_cnn_weights.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train_synthetic()
