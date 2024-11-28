# https://pytorch.org/tutorials/beginner/basics/intro.html

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from torch.utils.data import random_split
import os


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # TODO: Define the layers of the model
        # 1. Fully Connected Layers Only !!!
        # 2. Try different number of layers and neurons
        # 3. (Bonus) Try convolutional layers
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        # TODO: Define the forward pass of the model
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, train_loader, val_loader, optimizer, epoch, device):
    # Set the model to training mode
    model.train()
    train_loss = 0
    # TODO: Define the training loop
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        train_loss += loss.item()
        loss.backward()        
        optimizer.step()
        
        if batch_idx % args.log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                  f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
            if args.dry_run:
                break

    train_loss /= len(train_loader.dataset)
    
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)
    val_accuracy = 100. * correct / len(val_loader.dataset)

    print(f'\nEpoch {epoch} Summary:')
    print(f'Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')
    
    return train_loss, val_loss


def test(model, test_loader, device):
    # Set the model to evaluation mode
    model.eval()
    test_loss = 0
    correct = 0
    # TODO: Define the testing loop
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Test Average loss: {test_loss:.4f}, \
        Test Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)\n')
    return accuracy


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--model', type=str, default='Net', choices=['Net', 'CNN'],
                    help='Choose model architecture: Net or CNN (default: Net)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Detect GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # TODO: Tune the batch size to see different results
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if device.type == 'cuda':
        cuda_kwargs = {'num_workers': 2, 'pin_memory': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # Set transformation for the dataset
    # TODO: (Bonus) Change different dataset and transformations (data augmentation)
    # https://pytorch.org/vision/stable/datasets.html
    # https://pytorch.org/vision/main/transforms.html
    # e.g. CIFAR-10, Caltech101, etc. 
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)

    train_size = int(0.8 * len(dataset1))
    val_size = len(dataset1) - train_size
    train_dataset, val_dataset = random_split(dataset1, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    # Select model based on --model argument
    if args.model == 'Net':
        model = Net()
    elif args.model == 'CNN':
        model = CNN()
    model.to(device)  # Move model to GPU if available

    # TODO: Tune the learning rate / optimizer to see different results
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    # TODO: Tune the learning rate scheduler to see different results
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    
    train_losses, val_losses = [], []
    test_acc = 0

    for epoch in range(1, args.epochs + 1):
        # TODO: Return the loss and accuracy of the training loop and plot them
        # https://matplotlib.org/stable/tutorials/pyplot.html
        # train(args, model, train_loader, optimizer, epoch)
        # test(model, test_loader)
        train_loss, val_loss = train(args, model, train_loader, val_loader, optimizer, epoch, device)
        test_acc = test(model, test_loader, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step()

    # Plotting
    plt.figure(figsize=(8,6))
    plt.plot(range(1, args.epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, args.epochs + 1), val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Save to folder
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(os.path.join(output_dir, f'loss_{args.model}_batch_{args.batch_size}_epoch_{args.epochs}_lr_{args.lr}_testacc_{test_acc}.png'))

    if args.save_model:
        torch.save(model.state_dict(), "mnist.pt")

if __name__ == '__main__':
    main()
