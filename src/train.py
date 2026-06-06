import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from tqdm import tqdm
import random

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from src.cnn_deep import DeepCNN

DATASET_PATH = "dataset"
IMAGE_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 4
LEARNING_RATE = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Dispositivo: {DEVICE}")

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset = datasets.ImageFolder(DATASET_PATH, transform=transform)

num_classes = len(dataset.classes)
class_names = dataset.classes

print("\nCLASSES DO DATASET:")
for i, name in enumerate(class_names):
    print(f"{i}: {name}")

class_count = {name: 0 for name in class_names}

for _, label in dataset:
    class_count[class_names[label]] += 1

print("\nDISTRIBUIÇÃO DO DATASET:")
for k, v in class_count.items():
    print(f"{k}: {v} imagens")
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * (IMAGE_SIZE // 8) * (IMAGE_SIZE // 8), 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

def train_model(model, name="Model"):
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_weights = copy.deepcopy(model.state_dict())
    best_acc = 0

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(EPOCHS):

        print(f"\n[{name}] Epoch {epoch+1}/{EPOCHS}")

        model.train()

        total, correct, loss_sum = 0, 0, 0
        loop = tqdm(train_loader, desc=f"{name} Train", leave=False)

        for x, y in loop:
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)

            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            pred = out.argmax(1)

            correct += (pred == y).sum().item()
            total += y.size(0)

            loop.set_postfix(loss=loss.item())

        train_losses.append(loss_sum / len(train_loader))
        train_accs.append(100 * correct / total)

        model.eval()
        total, correct, loss_sum = 0, 0, 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)

                out = model(x)
                loss = criterion(out, y)

                loss_sum += loss.item()
                pred = out.argmax(1)

                correct += (pred == y).sum().item()
                total += y.size(0)

        val_losses.append(loss_sum / len(val_loader))
        val_accs.append(100 * correct / total)

        print(f"Train Acc: {train_accs[-1]:.2f}% | Val Acc: {val_accs[-1]:.2f}%")

        if val_accs[-1] > best_acc:
            best_acc = val_accs[-1]
            best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)

    return model, train_losses, val_losses, train_accs, val_accs

print("\nTREINANDO SIMPLE CNN...")
simple_model = SimpleCNN(num_classes)
simple_model, s_tl, s_vl, s_ta, s_va = train_model(simple_model, "SimpleCNN")

print("\nTREINANDO DEEP CNN...")
deep_model = DeepCNN(num_classes)
deep_model, d_tl, d_vl, d_ta, d_va = train_model(deep_model, "DeepCNN")

def test(model):
    model.eval()
    correct, total = 0, 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            out = model(x)
            pred = out.argmax(1)

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

            correct += (pred == y).sum().item()
            total += y.size(0)

    return 100 * correct / total, all_preds, all_labels

simple_test, s_pred, s_true = test(simple_model)
deep_test, d_pred, d_true = test(deep_model)

print("\n========================")
print(f"SimpleCNN Test Acc: {simple_test:.2f}%")
print(f"DeepCNN Test Acc: {deep_test:.2f}%")
print("========================")

def show_predictions(model, num=6):
    model.eval()

    images, labels = next(iter(test_loader))

    images = images.to(DEVICE)

    with torch.no_grad():
        outputs = model(images)
        preds = outputs.argmax(1)

    plt.figure(figsize=(12, 6))

    for i in range(num):
        img = images[i].cpu().permute(1, 2, 0)
        img = img * 0.5 + 0.5

        plt.subplot(2, 3, i+1)
        plt.imshow(img)
        plt.title(
            f"V: {class_names[labels[i]]}\nP: {class_names[preds[i]]}"
        )
        plt.axis("off")

    plt.show()

print("\nEXEMPLOS - SIMPLE CNN")
show_predictions(simple_model)

print("\nEXEMPLOS - DEEP CNN")
show_predictions(deep_model)

plt.figure(figsize=(10, 5))
plt.plot(s_tl, label="SimpleCNN")
plt.plot(d_tl, label="DeepCNN")
plt.title("Loss Comparison (SimpleCNN vs DeepCNN)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(s_ta, label="SimpleCNN")
plt.plot(d_ta, label="DeepCNN")
plt.title("Accuracy Comparison (SimpleCNN vs DeepCNN)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.show()
