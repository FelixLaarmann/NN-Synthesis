import torch 
import torch.nn as nn 

from experiments.dataset import load_mnist 

_NUM_CLASSES = 10 
_INPUT_SHAPE = (28, 28, 1)
_NUM_EPOCHS = 4
learning_rate = 0.01
batch_size = 64


class MnistExample(nn.Module):
    def __init__(self, in_channels):
        super(MnistExample, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=(3, 3), bias=False),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, kernel_size=(3, 3), bias=False),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(800, _NUM_CLASSES),
            nn.Softmax()
        )
    def forward(self, x):
        return self.layers(x)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MnistExample(_INPUT_SHAPE[-1]).to(device)


    (x_train, y_train), (x_test, y_test) = load_mnist()
    x_train = x_train.reshape(-1, 1, 28, 28)
    x_test = x_test.reshape(-1, 1, 28, 28)

    x_train_tensor = torch.tensor(x_train)
    y_train_tensor = torch.tensor(y_train)
    x_test_tensor = torch.tensor(x_test)
    y_test_tensor = torch.tensor(y_test)

    train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = torch.utils.data.TensorDataset(x_test_tensor, y_test_tensor)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.001, momentum = 0.9)

    for epoch in range(_NUM_EPOCHS):
        for i, (images, labels) in enumerate (train_loader):
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            loss = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print ('Epoch [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, _NUM_EPOCHS, loss.item()))
        
            # Validation
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                _, max_labels = torch.max(labels, 1)
                outputs = model(images)
                # print(outputs)
                _, predicted = torch.max(outputs.data, 1)
                # print(predicted)
                total += labels.size(0)
                correct += (predicted == max_labels).sum().item()
                del images, labels, outputs
        
            print('Accuracy of the network on the {} validation images: {} %'.format(5000, 100 * correct / total)) 
        