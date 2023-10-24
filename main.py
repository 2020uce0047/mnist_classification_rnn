import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

#Parameters
BATCH_SIZE = 32
input_size = 28
sequence_len = 28
num_layers = 2
hidden_size = 128
num_classes = 10
epochs = 5

# MNIST dataset 
train_dataset = torchvision.datasets.MNIST(root='./data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', 
                                           train=False,
                                           transform=transforms.ToTensor(),  
                                           download=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle = True)

#Model Architecture
class RNN(nn.Module):
    def __init__(self, batch_size, num_layers, input_size, hidden_size, num_classes):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first = True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h_0) #out -> [32, 28, 128]
        out = out[:, -1, :] #out -> [32, 128]
        return self.fc(out) #return -> [32, 10]

#Model, optimizer, criterion
model = RNN(BATCH_SIZE, num_layers, input_size, hidden_size, num_classes)
optimizer = Adam(model.parameters(), lr = 0.001)
criterion = CrossEntropyLoss()

#Training Loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    num_batches = 0
    for i, (images, labels) in enumerate(train_loader):
        x = images.view(-1, sequence_len, input_size) #images -> [32, 1, 28, 28]
        output = model(x)#x -> [32, 28, 28]

        optimizer.zero_grad()
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss
        num_batches += 1
    print(f'Epoch: {epoch+1} | Loss: {total_loss/num_batches}')

#Inference
model.eval()
with torch.no_grad():
    n_samples = 0
    correct_pred = 0
    for images, labels in test_loader:
        x = images.view(-1, sequence_len, input_size)
        output = model(x)
        _, predicted = torch.max(output, 1)
        n_samples += labels.size(0)
        correct_pred += (predicted == labels).sum()

print(f'Accuracy: {100*correct_pred/n_samples}')



