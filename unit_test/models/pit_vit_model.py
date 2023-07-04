import torch
import torch.nn as nn
import plinio.methods.pit.nn as pnn
from plinio.methods.pit.nn.features_masker import PITFeaturesMasker
from plinio.graph.features_calculation import ModAttrFeaturesCalculator, FeaturesCalculator
from plinio.methods.pit.nn.test import vit_to_pit
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
import timm
from timm.utils import accuracy
from timm.loss.cross_entropy import SoftTargetCrossEntropy
from timm.data.mixup import Mixup
from timm.data.random_erasing import RandomErasing
from timm.utils import accuracy
import os
from tqdm import tqdm

# train the model on cifar10
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# cifar10 stardard meand and deviation
img_mean = [0.4914, 0.4822, 0.4465]
img_std = [0.2023, 0.1994, 0.2010]
normalize = transforms.Normalize(mean=img_mean, std=img_std)
image_size = 384
patch_size = 16
train_transform = torchvision.transforms.Compose([transforms.RandAugment(2, 9), transforms.Resize(image_size), transforms.ToTensor(), normalize])
test_transform = torchvision.transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(), normalize])
train_ds = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
test_ds = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

batch_size = 32
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)

base_model = timm.create_model('vit_tiny_patch16_384', pretrained=True, num_classes=10)
model = vit_to_pit(base_model, (384,) * 2)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-2)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
criterion = SoftTargetCrossEntropy()


def evaluate(model, data_loader):
    model.eval()
    regulation_loss = 0
    loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(enumerate(data_loader), total=len(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)
            one_hot_target = torch.zeros_like(output).scatter_(1, target.unsqueeze(1), 1)
            loss += criterion(output, one_hot_target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            regulation_loss += model.get_size() 
        loss = loss / len(data_loader.dataset)
        correct /= len(data_loader.dataset)
        regulation_loss = regulation_loss / len(data_loader.dataset)
        return loss, correct, regulation_loss

def train(epoch, model, optimizer, scheduler, criterion, frequency=10):
    model.train()
    mixup = Mixup(mixup_alpha=0.8, cutmix_alpha=1.0, label_smoothing=0.1, num_classes=model.n_classes)
    random_erasing = RandomErasing(probability=0.25, mode='pixel', device = device)
    scaler = torch.cuda.amp.GradScaler()
    print(f"Epoch {epoch}")
    for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        data, target = data.to(device), target.to(device)
        data, target = mixup(data, target)
        data = random_erasing(data)
        with torch.cuda.amp.autocast():
            output = model(data)
            loss = criterion(output, target) + model.get_size()
        scaler.scale(loss).backward()
        # gradient accumulation
        if (batch_idx + 1) % frequency == 0 or batch_idx == len(train_loader) - 1:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
    scheduler.step()
 
loss_list = [] 
accuracy_list = []
regulation_loss_list = []

for epoch in range(20):
    train(epoch, model, optimizer, scheduler, criterion)
    loss, accuracy, regulation_loss = evaluate(model, train_loader)
    print(f"TRAIN loss: {loss}, accuracy: {accuracy}, regulation loss: {regulation_loss}")
    loss, accuracy, regulation_loss = evaluate(model, test_loader)
    print(f"TEST loss: {loss}, accuracy: {accuracy}, regulation loss: {regulation_loss}")
    loss_list.append(loss)
    accuracy_list.append(accuracy)
    regulation_loss_list.append(regulation_loss)

# plot the lists in three subplots
import matplotlib.pyplot as plt
fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
ax1.plot(loss_list)
ax1.set_title('loss')
ax2.plot(accuracy_list)
ax2.set_title('accuracy')
ax3.plot(regulation_loss_list)
ax3.set_title('regulation loss')
plt.tight_layout()
plt.show()

