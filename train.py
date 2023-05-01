import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import json
import numpy as np

from model import PNet, Licence

BS = 128
LR = 1e-3
EPOCHS = 20

def train(loader, net, loss_fn, optimizer, epoch, print_every=10):
    net.train()
    train_loss = 0
    for i, data in enumerate(loader):
        optimizer.zero_grad()
        X, labels = data
        out = net(X)
        loss = loss_fn(out, labels)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        if i % print_every == 0:
            print(f'Train Epoch: {epoch} [{i}/{len(loader)}]\tLoss: {loss.item():.6f}')
    net.eval()

def evaluate(loader, net):
    N_correct = 0
    with torch.no_grad():
        for data in loader:
            X, labels = data
            out = net(X)
            pred = torch.argmax(out, dim=1)
            N_correct += torch.sum(pred == torch.argmax(labels, dim=1))
    
    accuracy = N_correct / len(loader.dataset) * 100
    print(f'Test Accuracy: {N_correct}/{len(loader.dataset)} ({accuracy:.6f}%)')
    return accuracy

def initialize(N_classes, lr):
    net = PNet(N_classes)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    return net, loss_fn, optimizer
    
if __name__ == "__main__":
    base_dir = r"data\VehicleLicense\Data"
    dataset_tag = ["chinese", "chars"]
    for name in dataset_tag:
        with open(f"data\VehicleLicense\{name}_match.json", encoding="utf-8") as f:
            labels_map = json.load(f)
        torch.manual_seed(3407)
        np.random.seed(3407)
        # labels_map = {str(i): i for i in range(10)}
        dataset = Licence(base_dir, labels_map, True)
        train_size = int(len(dataset) * 0.95)
        test_size = len(dataset) - train_size
        
        train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
        train_loader = DataLoader(train_set, batch_size=BS, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=BS, shuffle=False)
        
        N_classes = len(labels_map)
        net, loss_fn, optimizer = initialize(N_classes, LR)
        
        for epoch in range(EPOCHS):
            train(train_loader, net, loss_fn, optimizer, epoch, 10)
        evaluate(test_loader, net)
        torch.save(net.state_dict(), f'checkpoints/{name}_Pnet.pt')

        