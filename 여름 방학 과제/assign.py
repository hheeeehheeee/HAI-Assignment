# ==============================
# GAN 기반 이미지 생성 모델 구현 (MNIST/FashionMNIST)
# ==============================

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os

# -----------------------------
# 1. Generator 정의
# -----------------------------


class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 28*28),
            nn.Tanh()   # 출력값 범위: [-1, 1]
        )

    def forward(self, z):
        return self.model(z).view(-1, 1, 28, 28)

# -----------------------------
# 2. Discriminator 정의
# -----------------------------


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()   # 진짜일 확률 반환
        )

    def forward(self, x):
        return self.model(x)


# -----------------------------
# 3. 데이터 전처리 및 로더
# -----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))   # [-1, 1] 정규화
])

# MNIST 사용 (FashionMNIST으로 바꾸려면 아래 주석 해제)
dataset = datasets.MNIST(root="./data", train=True,
                         download=True, transform=transform)
# dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)

loader = DataLoader(dataset, batch_size=128, shuffle=True)

# -----------------------------
# 4. 모델, 손실 함수, 옵티마이저
# -----------------------------
z_dim = 100
G = Generator(z_dim)
D = Discriminator()

criterion = nn.BCELoss()
opt_G = optim.Adam(G.parameters(), lr=0.0002)
opt_D = optim.Adam(D.parameters(), lr=0.0002)

# -----------------------------
# 5. 학습 루프
# -----------------------------
epochs = 200  # MNIST 기준 권장: 200~300
os.makedirs("samples", exist_ok=True)
fixed_noise = torch.randn(64, z_dim)

for epoch in range(1, epochs+1):
    for real, _ in loader:
        bs = real.size(0)

        # -----------------
        # (1) Discriminator 학습
        # -----------------
        noise = torch.randn(bs, z_dim)
        fake = G(noise)

        D_real = D(real)
        D_fake = D(fake.detach())

        loss_D = criterion(D_real, torch.ones_like(D_real)) + \
            criterion(D_fake, torch.zeros_like(D_fake))

        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # -----------------
        # (2) Generator 학습
        # -----------------
        D_fake = D(fake)
        loss_G = criterion(D_fake, torch.ones_like(D_fake))  # 가짜 이미지를 진짜로 속이기

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

    # -----------------
    # 에폭 결과 출력
    # -----------------
    print(
        f"[에폭 {epoch}/{epochs}] 판별자 손실: {loss_D.item():.4f} | 생성자 손실: {loss_G.item():.4f}")

    # 10 에폭마다 샘플 이미지 저장
    if epoch % 10 == 0:
        with torch.no_grad():
            fake = G(fixed_noise)
            save_image(
                fake, f"samples/epoch_{epoch}.png", nrow=8, normalize=True)
            print(f">>> 샘플 이미지 저장 완료: samples/epoch_{epoch}.png")
