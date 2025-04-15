# ------------------------
# 1. Imports and Dataset
# ------------------------
import os
import random
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch

# Custom dataset
class FlatFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.image_files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.folder_path, self.image_files[idx])
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, 0  # Dummy label

# Transform (normalize to [-1, 1])
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # for 3 channels
])

# Load datasets
satellite_dataset = FlatFolderDataset("./data/satellite", transform=transform)
map_dataset = FlatFolderDataset("./data/map", transform=transform)

# Data loaders
batch_size = 16
satellite_loader = DataLoader(satellite_dataset, batch_size=batch_size, shuffle=True)
map_loader = DataLoader(map_dataset, batch_size=batch_size, shuffle=True)

# Show a random satellite image
def show_random_satellite_sample(dataset):
    idx = random.randint(0, len(dataset) - 1)
    img_tensor, _ = dataset[idx]
    img = img_tensor.permute(1, 2, 0) * 0.5 + 0.5  # Unnormalize
    plt.imshow(img)
    plt.title("Sample Satellite Image")
    plt.axis("off")
    plt.show()

show_random_satellite_sample(satellite_dataset)
