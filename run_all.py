import torch

from src.cnn_deep import DeepCNN
from src.train import train_model
from src.evaluate import evaluate_model
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_PATH = "dataset"
IMAGE_SIZE = 64
BATCH_SIZE = 32

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset = datasets.ImageFolder(DATASET_PATH, transform=transform)

class_names = dataset.classes
num_classes = len(class_names)

train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_ds, val_ds, test_ds = random_split(
    dataset,
    [train_size, val_size, test_size]
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("INICIANDO TREINO DO MODELO E GERANDO GRÁFICO DE MATRIZ: deep_cnn")
    print("=" * 60)

    model = DeepCNN(num_classes).to(device)

    train_model(model, name="deep_cnn")

    print("\nRESULTADOS NO TEST SET:")

    evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        class_names=class_names
    )

    torch.save(model.state_dict(), "models/deep_cnn.pth")

    print("\nModelo salvo com sucesso!")
    print("\nTREINAMENTO FINALIZADO!")
