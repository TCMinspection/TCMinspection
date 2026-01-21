# CycleGAN: Yellow → Cyan (Blue-Green) Face Domain Translation
# Author: ChatGPT for Zubin's TCM Face Color Classification Project

import os
import random
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
from torchvision.utils import save_image

# ----------------------------
# 1. Dataset Preparation
# ----------------------------

class UnpairedFaceDataset(Dataset):
    def __init__(self, dir_yellow, dir_cyan, transform=None):
        self.yellow_imgs = list(Path(dir_yellow).glob("*.jpg")) + list(Path(dir_yellow).glob("*.png"))
        self.cyan_imgs = list(Path(dir_cyan).glob("*.jpg")) + list(Path(dir_cyan).glob("*.png"))
        self.transform = transform

    def __len__(self):
        return max(len(self.yellow_imgs), len(self.cyan_imgs))

    def __getitem__(self, index):
        yellow_img = Image.open(self.yellow_imgs[index % len(self.yellow_imgs)]).convert("RGB")
        cyan_img = Image.open(self.cyan_imgs[random.randint(0, len(self.cyan_imgs) - 1)]).convert("RGB")
        if self.transform:
            yellow_img = self.transform(yellow_img)
            cyan_img = self.transform(cyan_img)
        return {"yellow": yellow_img, "cyan": cyan_img}

class SingleDomainDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = list(Path(image_dir).glob("*.jpg")) + list(Path(image_dir).glob("*.png"))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, img_path.name

# ----------------------------
# 2. Generator and Discriminator
# ----------------------------

class ResidualBlock(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3),
            nn.InstanceNorm2d(features)
        )

    def forward(self, x):
        return x + self.block(x)

class GeneratorResNet(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super().__init__()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]
        in_features = 64
        out_features = in_features * 2
        for _ in range(3):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features *= 2
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]
        out_features = in_features // 2
        for _ in range(3):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features //= 2
        model += [nn.ReflectionPad2d(3), nn.Conv2d(64, output_nc, 7), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super().__init__()
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(input_nc, 64, normalize=False),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, x):
        return self.model(x)

# ----------------------------
# 3. Training Loop
# ----------------------------

def train_cyclegan(data_loader, generator_Y2C, generator_C2Y, discriminator_Y, discriminator_C,
                criterion_GAN, criterion_cycle, optimizer_G, optimizer_D_Y, optimizer_D_C,
                device, num_epochs=100, save_dir="checkpoints"):
    os.makedirs(save_dir, exist_ok=True)
    for epoch in range(num_epochs):
        for i, batch in enumerate(tqdm(data_loader)):
            real_Y = batch["yellow"].to(device)
            real_C = batch["cyan"].to(device)

            valid = torch.ones_like(discriminator_Y(real_Y)).to(device)
            fake = torch.zeros_like(valid)

            optimizer_G.zero_grad()
            fake_C = generator_Y2C(real_Y)
            rec_Y = generator_C2Y(fake_C)
            fake_Y = generator_C2Y(real_C)
            rec_C = generator_Y2C(fake_Y)

            loss_GAN_Y2C = criterion_GAN(discriminator_C(fake_C), valid)
            loss_GAN_C2Y = criterion_GAN(discriminator_Y(fake_Y), valid)

            loss_cycle_Y = criterion_cycle(rec_Y, real_Y)
            loss_cycle_C = criterion_cycle(rec_C, real_C)

            loss_G = loss_GAN_Y2C + loss_GAN_C2Y + 10.0 * (loss_cycle_Y + loss_cycle_C)
            loss_G.backward()
            optimizer_G.step()

            optimizer_D_Y.zero_grad()
            loss_real = criterion_GAN(discriminator_Y(real_Y), valid)
            loss_fake = criterion_GAN(discriminator_Y(fake_Y.detach()), fake)
            loss_D_Y = 0.5 * (loss_real + loss_fake)
            loss_D_Y.backward()
            optimizer_D_Y.step()

            optimizer_D_C.zero_grad()
            loss_real = criterion_GAN(discriminator_C(real_C), valid)
            loss_fake = criterion_GAN(discriminator_C(fake_C.detach()), fake)
            loss_D_C = 0.5 * (loss_real + loss_fake)
            loss_D_C.backward()
            optimizer_D_C.step()

            if i % 100 == 0:
                print(f"Epoch [{epoch}/{num_epochs}], Step [{i}], G Loss: {loss_G.item():.4f}")

        save_sample_images(generator_Y2C, generator_C2Y, real_Y, real_C, epoch, device)

        torch.save(generator_Y2C.state_dict(), os.path.join(save_dir, f"G_Y2C_epoch{epoch}.pth"))
        torch.save(generator_C2Y.state_dict(), os.path.join(save_dir, f"G_C2Y_epoch{epoch}.pth"))
        torch.save(discriminator_Y.state_dict(), os.path.join(save_dir, f"D_Y_epoch{epoch}.pth"))
        torch.save(discriminator_C.state_dict(), os.path.join(save_dir, f"D_C_epoch{epoch}.pth"))

# ----------------------------
# 4. Save Visual Results
# ----------------------------

def save_sample_images(G_Y2C, G_C2Y, real_Y, real_C, epoch, device, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    G_Y2C.eval()
    G_C2Y.eval()
    with torch.no_grad():
        fake_C = G_Y2C(real_Y[:4])
        fake_Y = G_C2Y(real_C[:4])
        combined = torch.cat([real_Y[:4], fake_C, real_C[:4], fake_Y], dim=0)
        grid = transforms.ToPILImage()(make_grid(combined.cpu(), nrow=4, normalize=True))
        grid.save(f"{save_dir}/epoch_{epoch:03d}.png")
    G_Y2C.train()
    G_C2Y.train()

# ----------------------------
# 5. Image Sorting Utility
# ----------------------------

def sort_images_by_label(input_dir, output_root):
    allowed_labels = {"red", "black", "white", "yellow", "cyan"}
    os.makedirs(output_root, exist_ok=True)
    for file in Path(input_dir).glob("*.*"):
        if file.suffix.lower() not in [".jpg", ".png", ".jpeg"]:
            continue
        parts = file.stem.split('_')
        if len(parts) < 2:
            continue
        label = parts[-1].lower()
        if label not in allowed_labels:
            continue
        dest_dir = Path(output_root) / label
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / file.name
        os.rename(file, dest_path)

# ----------------------------
# 6. Inference Function
# ----------------------------

def inference_yellow_to_cyan(model_path, yellow_dir, output_dir, image_size, device):
    os.makedirs(output_dir, exist_ok=True)
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = SingleDomainDataset(yellow_dir, transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = GeneratorResNet(3, 3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        for img, name in tqdm(loader):
            img = img.to(device)
            fake = model(img)
            fake = 0.5 * (fake + 1.0)
            save_image(fake, os.path.join(output_dir, name[0]))

# ----------------------------
# 7. Main Entry Point
# ----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "inference"], default="train")
    parser.add_argument("--yellow_dir", type=str, default="D:/TCMinspection/imagesource1/sorted_faces/yellow")
    parser.add_argument("--cyan_dir", type=str, default="D:/TCMinspection/imagesource1/sorted_faces/cyan")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/G_Y2C_epoch49.pth")
    parser.add_argument("--output_dir", type=str, default="./inference_output")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "train":
        transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = UnpairedFaceDataset(args.yellow_dir, args.cyan_dir, transform)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

        G_Y2C = GeneratorResNet(3, 3).to(device)
        G_C2Y = GeneratorResNet(3, 3).to(device)
        D_Y = Discriminator(3).to(device)
        D_C = Discriminator(3).to(device)

        criterion_GAN = nn.MSELoss()
        criterion_cycle = nn.L1Loss()

        optimizer_G = torch.optim.Adam(
            list(G_Y2C.parameters()) + list(G_C2Y.parameters()), lr=0.0002, betas=(0.5, 0.999))
        optimizer_D_Y = torch.optim.Adam(D_Y.parameters(), lr=0.0002, betas=(0.5, 0.999))
        optimizer_D_C = torch.optim.Adam(D_C.parameters(), lr=0.0002, betas=(0.5, 0.999))

        train_cyclegan(dataloader, G_Y2C, G_C2Y, D_Y, D_C,
                    criterion_GAN, criterion_cycle,
                    optimizer_G, optimizer_D_Y, optimizer_D_C,
                    device, num_epochs=args.num_epochs)

    elif args.mode == "inference":
        inference_yellow_to_cyan(
            model_path=args.checkpoint,
            yellow_dir=args.yellow_dir,
            output_dir=args.output_dir,
            image_size=args.image_size,
            device=device
        )
