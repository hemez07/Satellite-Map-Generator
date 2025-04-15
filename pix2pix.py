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




# ------------------------
# 2. Model Definitions
# ------------------------
import torch.nn as nn

# Generator (Very basic)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, 128, 128]
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # [B, 3, 256, 256]
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Discriminator (PatchGAN-style with LazyLinear to avoid shape mismatch)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, 128, 128]
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # [B, 128, 64, 64]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.LazyLinear(1),  # No need to hardcode input shape
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Initialize models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G_sat2map = Generator().to(device)
D_map = Discriminator().to(device)





# ------------------------
# 3. Training Loop
# ------------------------
import torch.optim as optim

# Loss functions
adversarial_loss = nn.BCELoss()
l1_loss = nn.L1Loss()

# Optimizers
lr = 0.0002
optimizer_G = optim.Adam(G_sat2map.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D_map.parameters(), lr=lr, betas=(0.5, 0.999))

# Training
epochs = 10
for epoch in range(epochs):
    for (sat_images, _), (map_images, _) in zip(satellite_loader, map_loader):
        sat_images = sat_images.to(device)
        map_images = map_images.to(device)

        # Generate fake maps
        fake_map = G_sat2map(sat_images)

        # ------------------
        # Train Discriminator
        # ------------------
        real_preds = D_map(map_images)
        real_loss = adversarial_loss(real_preds, torch.ones_like(real_preds))

        fake_preds = D_map(fake_map.detach())
        fake_loss = adversarial_loss(fake_preds, torch.zeros_like(fake_preds))

        d_loss = (real_loss + fake_loss) / 2
        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # ------------------
        # Train Generator
        # ------------------
        g_adv = adversarial_loss(D_map(fake_map), torch.ones_like(fake_preds))
        g_l1 = l1_loss(fake_map, map_images)
        total_g_loss = g_adv + 10 * g_l1

        optimizer_G.zero_grad()
        total_g_loss.backward()
        optimizer_G.step()

    print(f"Epoch [{epoch+1}/{epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {total_g_loss.item():.4f}")
