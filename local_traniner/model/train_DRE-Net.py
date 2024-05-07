import os

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay
from torch import nn, optim
from torch.utils.data import DataLoader

from local_traniner.model.core.dataset import CovidDataSet
from local_traniner.model.core.model import DRE_net

model_path = 'DRE-Net.pth'

BATCH_SIZE = 4
EPOCHS = 30

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')
    print()

train_path = os.path.join('..', 'input', 'train')
val_path = os.path.join('..', 'input', 'val')
test_path = os.path.join('..', 'input', 'test')

train_data = CovidDataSet(root=train_path, is_train=True)
val_data = CovidDataSet(root=val_path, is_train=False)
test_data = CovidDataSet(root=test_path, is_train=False)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=False)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, drop_last=False)
test_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, drop_last=False)


def calculate_accuracy(dataloader: DataLoader, model: nn.Module):
    correct = 0.
    total = 0.

    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, pred = torch.max(outputs, 1)

            correct += (pred == labels).sum().cpu().numpy()
            total += labels.size(0)

    return 100 * correct / total


def roc_curve(dataloader: DataLoader, model: nn.Module):
    y_true = []
    y_scores = []

    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            y_score = torch.softmax(outputs, dim=1)[:, 1]

            y_true.extend(labels.cpu().numpy())
            y_scores.extend(y_score.cpu().numpy())

    RocCurveDisplay.from_predictions(y_true, y_scores, pos_label=1, name='DRE-net')
    plt.title('ROC Curve')
    plt.show()


def confusion_matrix(dataloader: DataLoader, model: nn.Module):
    y_true = []
    y_pred = []

    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, pred = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=['noCovid', 'Covid'])
    plt.title('Confusion Matrix')
    plt.show()


if __name__ == "__main__":
    model = DRE_net(n_class=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.002, momentum=0.9, weight_decay=1e-4)

    # checkpoint = torch.load(model_path)
    # epoch = checkpoint['epoch']
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # loss = checkpoint['loss']
    # val_accuracy = checkpoint['val_accuracy']

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {running_loss / len(train_loader)}")

        if (epoch + 1) % 5 == 0:
            val_accuracy = calculate_accuracy(val_loader, model)
            print(f"Accuracy of the network on the val images: {val_accuracy:.1f} %")

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': running_loss,
                'val_accuracy': val_accuracy
            }, model_path)

    test_accuracy = calculate_accuracy(test_loader, model)
    print(f"Accuracy of the final network on the test images: {test_accuracy:.1f} %")
    roc_curve(test_loader, model)
    confusion_matrix(test_loader, model)
