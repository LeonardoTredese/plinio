import torch
import torch.nn as nn
import plinio.methods.pit.nn as pnn
from plinio.methods.pit.nn.features_masker import PITFeaturesMasker
from plinio.graph.features_calculation import ModAttrFeaturesCalculator, FeaturesCalculator
from plinio.methods.pit.nn.test import vit_to_pit
from plinio.methods.pit.nn import PITConv2d, PITLinear
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
import wandb

# train the model on cifar10
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# configuration
config = {
    'base_model': 'vit_tiny_patch16_384',
    'dataset': 'cifar10',
    'batch_size': 128,
    'learning_rate': 1e-4,
    'weight_decay': 5e-2,
    'epochs': 500,
    'size_lambda': 1e-8,
    'weight_update_frequency': 10,
    'image_size': 384,
    'patch_size': 16,
}



# cifar10 stardard meand and deviation
img_mean = [0.4914, 0.4822, 0.4465]
img_std = [0.2023, 0.1994, 0.2010]
normalize = transforms.Normalize(mean=img_mean, std=img_std)
image_size = config['image_size']
patch_size = config['patch_size']
train_transform = torchvision.transforms.Compose([transforms.RandAugment(2, 9), transforms.Resize(image_size), transforms.ToTensor(), normalize])
test_transform = torchvision.transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(), normalize])
train_ds = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
test_ds = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

batch_size = config['batch_size']
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)


base_model = timm.create_model(config['base_model'], pretrained=True, num_classes=10)
model = vit_to_pit(base_model, (384,) * 2).to(device)
optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=4, T_mult=1, eta_min=1e-8)
criterion = SoftTargetCrossEntropy()

# set the nas parameters to the binarizer threshold
with torch.no_grad():
    threshold = 0.5
    for p in model.nas_parameters():
        p.fill_(threshold + 1e-3)

def evaluate(model, data_loader):
    model.eval()
    regulation_loss = 0
    loss = 0
    correct = 0
    print("Evaluating...")
    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(enumerate(data_loader), total=len(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)
            one_hot_target = torch.zeros_like(output).scatter_(1, target.unsqueeze(1), 1)
            loss += criterion(output, one_hot_target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        loss = loss / len(data_loader.dataset)
        correct /= len(data_loader.dataset)
        return loss, correct, model.get_size()

def train(epoch, model, optimizer, scheduler, criterion, size_lambda, frequency=10):
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
            loss = criterion(output, target) + size_lambda * model.get_size()
        scaler.scale(loss).backward()
        # gradient accumulation
        if (batch_idx + 1) % frequency == 0 or batch_idx == len(train_loader) - 1:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
    scheduler.step()

def add_prefix(prefix, d):
    return {prefix + key: value for key, value in d.items()}

# initialize wandb
wandb.init(project='pit-vit', config=config)
wandb.watch(model)

for epoch in range(config['epochs']):
    train(epoch, model, optimizer, scheduler, criterion, size_lambda=config['size_lambda'], frequency=config['weight_update_frequency'])
    train_loss, train_accuracy, _ = evaluate(model, train_loader)
    print(f"TRAIN loss: {train_loss}, accuracy: {train_accuracy}")
    test_loss, test_accuracy, model_size = evaluate(model, test_loader)
    print(f"TEST loss: {test_loss}, accuracy: {test_accuracy}, model size: {model_size}")
    metrics = {'train_loss': train_loss, 'train_accuracy': train_accuracy, 'test_loss': test_loss, 'test_accuracy': test_accuracy, 'model_size': model_size}
    layers_params = dict()
    for name, param in model.named_modules():
        if isinstance(param, PITConv2d) or isinstance(param, PITLinear):
            layers_params[f"{name}"] = param.out_features_opt
    wandb.log(add_prefix("metrics/", metrics) | add_prefix("layers_masks/", layers_params))
wandb.finish()
