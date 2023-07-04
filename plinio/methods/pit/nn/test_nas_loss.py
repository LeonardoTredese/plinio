import torch
import torch.nn as nn
from plinio.methods.pit.nn.test import vit_to_pit
import torch.optim as optim
import timm
import os
from tqdm import tqdm

# train the model on cifar10
device = 'cuda' if torch.cuda.is_available() else 'cpu'


batch_size = 64
size = (384,) * 2
channels = 3


base_model = timm.create_model('vit_tiny_patch16_384', pretrained=True, num_classes=10)
model = vit_to_pit(base_model, (384,) * 2)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-2)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)


def train(epoch, model, optimizer, scheduler):
    model.train()
    print(f"Epoch {epoch}")
    optimizer.zero_grad()
    data = torch.randn(batch_size, channels, *size).to(device)
    data = data.to(device) 
    output = model(data)
    loss = model.get_size()
    loss.backward()
    optimizer.step()
    scheduler.step()
 
sizes = []
lr_list = []

for epoch in range(20):
    train(epoch, model, optimizer, scheduler)
    with torch.no_grad():
        print(f"Epoch {epoch}")
        print(f"Size: {model.get_size()}")
        sizes.append(model.get_size())
        lr_list.append(scheduler.get_last_lr()[0])

# plot the size and the learning rate over the epochs in different subplots
import matplotlib.pyplot as plt
fig, axs = plt.subplots(2)
axs[0].plot(sizes)
axs[0].set_title("Size")
axs[1].plot(lr_list)
axs[1].set_title("Learning rate")
plt.show()
