import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as AF
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


### CNN architecture
class CNN(nn.Module):
    def __init__(self, dropout_pr, num_hidden, num_classes):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout_conv1 = nn.Dropout2d(dropout_pr)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout_conv2 = nn.Dropout2d(dropout_pr)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout_conv3 = nn.Dropout2d(dropout_pr)

        self.num_flatten_nodes = 128 * 3 * 3
        self.fc1 = nn.Linear(self.num_flatten_nodes, num_hidden)
        self.out = nn.Linear(num_hidden, num_classes)

    def forward(self, x):
        out = AF.relu(self.pool1(self.conv1(x)))
        out = AF.relu(self.dropout_conv1(out))

        out = AF.relu(self.pool2(self.conv2(out)))
        out = AF.relu(self.dropout_conv2(out))

        out = AF.relu(self.pool3(self.conv3(out)))
        out = AF.relu(self.dropout_conv3(out))

        out = out.view(-1, self.num_flatten_nodes)
        out = AF.relu(self.fc1(out))
        out = AF.dropout(out)
        output = self.out(out)

        return output


# Training function
def train_CNN_model(num_epochs, training_data, device, CNN_model, loss_func, optimizer):
    train_losses = []
    CNN_model.train()
    for epoch_cnt in range(num_epochs):
        for batch_cnt, (images, labels) in enumerate(training_data):
            if (device.type == 'cuda'):
                images = images.to(device)
                labels = labels.to(device)

            optimizer.zero_grad()
            output = CNN_model(images)
            loss = loss_func(output, labels)
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()

            if (batch_cnt + 1) % mini_batch_size == 0:
                print(f"Epoch={epoch_cnt + 1}/{num_epochs}, batch={batch_cnt + 1}/{num_train_batches}, loss={loss.item()}")
    return train_losses


# Testing function
def test_CNN_model(device, CNN_model, testing_data):
    predicted_digits = []
    with torch.no_grad():
        CNN_model.eval()
        for batch_cnt, (images, labels) in enumerate(testing_data):
            if (device.type == 'cuda'):
                images = images.to(device)
                labels = labels.to(device)

            output = CNN_model(images)
            _, prediction = torch.max(output, 1)
            predicted_digits.append(prediction)
            num_samples = labels.shape[0]
            num_correct = (prediction == labels).sum().item()
            accuracy = num_correct / num_samples
            if (batch_cnt + 1) % mini_batch_size == 0:
                print(f"batch={batch_cnt + 1}/{num_test_batches}")
        print("> Number of samples=", num_samples, "number of correct prediction=", num_correct, "accuracy=", accuracy)
    return predicted_digits


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print("------------------ANN modeling---------------------------")
    transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), ])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms, download=False)
    print("> Shape of training data:", train_dataset.data.shape)
    print("> Shape of testing data:", test_dataset.data.shape)
    print("> Classes:", train_dataset.classes)

    mini_batch_size = 100
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=mini_batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=mini_batch_size, shuffle=True)
    num_train_batches = len(train_dataloader)
    num_test_batches = len(test_dataloader)
    print("> Mini batch size: ", mini_batch_size)
    print("> Number of batches loaded for training: ", num_train_batches)
    print("> Number of batches loaded for testing: ", num_test_batches)

    num_classes = 10
    num_hidden = 128
    dropout_pr = 0.05

    # CNN model
    CNN_model = CNN(dropout_pr, num_hidden, num_classes)
    print("> CNN model parameters")
    print(CNN_model.parameters)

    if (device.type == 'cuda'):
        print("...Modeling using GPU...")
        CNN_model = CNN_model.to(device=device)
    else:
        print("...Modeling using CPU...")

    loss_func = nn.CrossEntropyLoss()
    num_epochs = 1
    alpha = 0.001
    CNN_optimizer = optim.Adam(CNN_model.parameters(), lr=alpha)

    print("............Training CNN................")
    train_loss = train_CNN_model(num_epochs, train_dataloader, device, CNN_model, loss_func, CNN_optimizer)
    print("............Testing CNN model................")
    predicted_digits = test_CNN_model(device, CNN_model, test_dataloader)
