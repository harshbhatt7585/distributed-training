import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

def train(rank, args):
    print(f"[Process {rank}] Starting training")
    
    model = SimpleModel()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    x = torch.randn(100, 10)
    y = torch.randn(100, 1)

    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=10, shuffle=True)

    for epoch in range(3):
        for batch_x, batch_y in loader:
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"[Process {rank}] Epoch {epoch+1}, Loss: {loss.item():.4f}")

def main():
    world_size = 4
    mp.spawn(train, args=(None,), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()
