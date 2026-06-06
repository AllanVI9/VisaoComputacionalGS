import sys
import torch
from PIL import Image
from torchvision import transforms

from cnn_deep import DeepCNN

IMAGE_SIZE = 64

CLASS_NAMES = [
    "Agriculture",
    "Forest",
    "Residential",
    "Water"
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

model = DeepCNN(
    num_classes=len(CLASS_NAMES)
).to(DEVICE)

model.load_state_dict(
    torch.load(
        "models/best_model.pth",
        map_location=DEVICE
    )
)

model.eval()

image_path = sys.argv[1]

image = Image.open(
    image_path
).convert("RGB")

image = transform(image)

image = image.unsqueeze(0)

image = image.to(DEVICE)

with torch.no_grad():

    outputs = model(image)

    probabilities = torch.softmax(
        outputs,
        dim=1
    )

    confidence, prediction = torch.max(
        probabilities,
        1
    )

predicted_class = CLASS_NAMES[
    prediction.item()
]

confidence = confidence.item() * 100

print("\nResultado:")
print(f"Classe: {predicted_class}")
print(f"Confiança: {confidence:.2f}%")
