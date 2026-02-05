import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from python_polar_coding.channels import SimpleBPSKModulationAWGN
from python_polar_coding.polar_codes.sc_list.codec import SCListPolarCodec
from python_polar_coding.simulation.functions import generate_binary_message
from python_polar_coding.polar_codes.ai_scl.decoding_path import PathPruningNet

class PolarCodeDataset(Dataset):
    """Generate synthetic polar code data for training."""
    def __init__(self, N, K, num_samples, snr_db):
        self.N = N
        self.K = K
        self.num_samples = num_samples
        self.snr_db = snr_db
        self.codec = SCListPolarCodec(N=N, K=K, design_snr=0.0, L=8)
        self.bpsk = SimpleBPSKModulationAWGN(fec_rate=K/N)
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random message
        msg = generate_binary_message(size=self.K)
        
        # Encode
        encoded = self.codec.encode(msg)
        
        # Transmit over AWGN channel
        transmitted = self.bpsk.transmit(message=encoded, snr_db=self.snr_db)
        
        # LLR is the received signal scaled
        llr = 2 * transmitted / (self.bpsk.noise_power)
        
        # Label: 1 if correct bits, 0 if not (binary classification)
        # We'll use the ground truth message as label
        label = torch.tensor(msg, dtype=torch.float32)
        
        return torch.tensor(llr, dtype=torch.float32), label

class PathScoringNet(nn.Module):
    """Neural network for scoring decoding paths."""
    def __init__(self, N):
        super().__init__()
        self.fc1 = nn.Linear(2 * N, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))  # Output between 0 and 1
        return x

def train_model(N, K, num_epochs=20, snr_db=1.0, batch_size=32):
    """Train the PathPruningNet."""
    print(f"Training PathPruningNet for ({N}, {K}) polar code")
    print(f"SNR: {snr_db} dB, Epochs: {num_epochs}, Batch size: {batch_size}")
    
    # Create dataset and dataloader
    dataset = PolarCodeDataset(N, K, num_samples=1000, snr_db=snr_db)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create model
    model = PathScoringNet(N)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        for llr_batch, label_batch in dataloader:
            llr_batch = llr_batch.to(device)
            label_batch = label_batch.to(device)
            
            # Forward pass: concatenate LLR with zero bits (initial state)
            bits_batch = torch.zeros_like(llr_batch)
            x = torch.cat([llr_batch, bits_batch], dim=1)
            
            # Predict correctness of path (simplified: predict avg bit correctness)
            output = model(x)
            
            # Target: use average correctness of message
            target = label_batch.mean(dim=1, keepdim=True)
            
            loss = criterion(output, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
    
    print("Training complete!")
    return model

def save_model(model, filepath):
    """Save trained model."""
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")

def load_model(N, filepath):
    """Load trained model."""
    model = PathScoringNet(N)
    model.load_state_dict(torch.load(filepath))
    model.eval()
    print(f"Model loaded from {filepath}")
    return model

if __name__ == "__main__":
    N = 128
    K = 64
    model = train_model(N, K, num_epochs=20, snr_db=1.0, batch_size=32)
    save_model(model, f"trained_model_N{N}_K{K}.pt")
