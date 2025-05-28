import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

transform = transforms.Compose([transforms.Resize(
    (28, 28)), transforms. ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])


train_ds = datasets.MNIST(root='data', train=True,
                          download=True, transform=transform)
test_ds = datasets.MNIST(root='data', train=False,
                         download=True, transform=transform)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=1000, shuffle=False)


class myModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.model(x)


model = myModel()

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 10
for epoch in range(1, epochs+1):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch}, Loss: {total_loss/len(train_loader)}")

img = Image.open(
    '/Users/jeonghs215/HAI-Assignment/딥러닝기초반6주차/numberImage.png').convert('L')
to_tensor = transforms.Compose([transforms.Resize(
    (28, 28)), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

model.eval()
x = to_tensor(img).unsqueeze(0)
with torch.no_grad():
    logits = model(x)
    probs = nn.functional.softmax(logits, dim=1)

print(probs)
