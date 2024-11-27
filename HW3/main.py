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
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device: {device}\n")

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


def train(args, model, train_loader, optimizer, epoch):
    # Set the model to training mode
    model.train()
    train_loss = 0
    correct = 0
    # TODO: Define the training loop
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f'Train Epoch {epoch}')):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        
        # if batch_idx % args.log_interval == 0:
        #     print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
        #           f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        #     if args.dry_run:
        #         break
    
    train_loss /= len(train_loader.dataset)
    accuracy = 100. * correct / len(train_loader.dataset)
    print(f'Train set: Average loss: {train_loss:.4f}, Accuracy: {accuracy:.2f}%')
    return train_loss, accuracy


def test(model, test_loader):
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

    # Log the testing status
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)    
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%\n')
    return test_loss, accuracy



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
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # TODO: Tune the batch size to see different results
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}

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
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net()
    # TODO: Tune the learning rate / optimizer to see different results
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    # TODO: Tune the learning rate scheduler to see different results
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    for epoch in range(1, args.epochs + 1):
        # TODO: Return the loss and accuracy of the training loop and plot them
        # https://matplotlib.org/stable/tutorials/pyplot.html
        train_loss, train_acc = train(args, model, train_loader, optimizer, epoch)
        test_loss, test_acc = test(model, test_loader)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        scheduler.step()
        # train(args, model, train_loader, optimizer, epoch)
        # test(model, test_loader)
        # scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist.pt")

    # Plotting results
    # plt.figure(figsize=(12, 4))
    # plt.subplot(1, 2, 1)
    plt.plot(range(1, args.epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, args.epochs + 1), test_losses, label='Test Loss')
    # plt.ylim((0, 0.6))
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Loss Curve')

    # plt.subplot(1, 2, 2)
    # plt.plot(range(1, args.epochs + 1), train_accuracies, label='Train Accuracy')
    # plt.plot(range(1, args.epochs + 1), test_accuracies, label='Test Accuracy')
    # plt.legend()
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy (%)')
    # plt.title('Accuracy Curve')

    # plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
