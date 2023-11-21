import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms


# =========================== working with data =====================
fashion_train_data = datasets.FashionMNIST(
    "./data/fashion_mnist", True, transform=transforms.ToTensor(), download=True)
fashion_test_data = datasets.FashionMNIST(
    "./data/fashion_mnist", False,  transform=transforms.ToTensor(), download=True)

x, y = fashion_test_data[0]
print(type(x), type(y))

batch_size = 8
train_dataloader = DataLoader(
    fashion_train_data, batch_size=batch_size, shuffle=True, num_workers=4)
test_dataloader = DataLoader(
    fashion_test_data, batch_size=batch_size, shuffle=True, num_workers=4)

x, y = next(iter(train_dataloader))
print(x.shape, y.shape)


# =========================== creating Models =====================
device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")


class NeuralNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device=device)
print(model)


# =========================== loss function =====================
loss_fn = nn.CrossEntropyLoss()


# =========================== optimizer =====================
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


# =========================== train =====================
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        y_hat = model(x)
        loss = loss_fn(y_hat, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 1000 == 0:
            print(f"loss:{loss.item():>7f} [{batch*len(x):>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            y_hat: torch.Tensor = model(x)
            test_loss += loss_fn(y_hat, y)
            correct += (y_hat.argmax(dim=1) == y).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: Accuracy:{(correct*100):>0.1f}%, Avg loss:{test_loss:>8f}")


epochs = 1
for t in range(epochs):
    print("-"*20+f"Epoch {t+1}"+"-"*20)
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")


# =========================== save model =====================

model_pth_path = "./tmp/model.pth"
torch.save(model.state_dict(), model_pth_path)

# =========================== load model =====================
model1 = NeuralNetwork().to(device)
model1.load_state_dict(torch.load(model_pth_path))

# =========================== inference ======================

CLASSED = ["T-shirt/top",
           "Trouser",
           "Pullover",
           "Dress",
           "Coat",
           "Sandal",
           "Shirt",
           "Sneaker",
           "Bag",
           "Ankle boot"]
with torch.no_grad():
    x, y = fashion_test_data[0]
    x = x.unsqueeze_(dim=0).to(device)
    y_hat: torch.Tensor = model(x)
    print(f"predict: {CLASSED[y_hat.argmax(1)[0]]}, label: {CLASSED[y]}")
