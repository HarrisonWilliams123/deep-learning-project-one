#Imports the libraries for a simple deep learning model
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

#Creates the training and test data split and initializes the dataloaders
training_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

#Neural Network Class
class NeuralNetwork(nn.Module):
    #Initializes the Neural Network
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10),
        )
    #Forward propagation method
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits 

model = NeuralNetwork()

#Hyperparameters for the Neural Network
learning_rate = 1e-3
batch_size = 64
epochs = 5

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    #Sets the model to training mode
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        #Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        #Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
