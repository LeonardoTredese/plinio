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
import pytorch_benchmarks.transformers.image_classification as icl
from pytorch_benchmarks.utils import seed_all, EarlyStopping
import os
from tqdm import tqdm
import wandb

# train the model on cifar10
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ensure deterministic behavior
seed = seed_all(seed=42)

# configuration
config = {
    'base_model': 'vit_tiny_patch16_224',
    'dataset': 'cifar10',
    'batch_size': 128,
    'learning_rate': 1e-4,
    'nas_learning_rate': 1e-2,
    'weight_decay': 5e-2,
    'epochs': 500,
    'size_lambda': 1e-8,
    'weight_update_frequency': 10,
    'image_size': 224,
    'patch_size': 16,
    'checkpoint_frequency': 10,
}



# cifar10 stardard meand and deviation
image_size = config['image_size']
patch_size = config['patch_size']
datasets = icl.get_data(dataset=config['dataset'], download=True, image_size=(image_size, image_size))
dataloaders = icl.build_dataloaders(datasets, batch_size=config['batch_size'], num_workers=os.cpu_count(), seed=seed)
train_dl, val_dl, test_dl = dataloaders

base_model = timm.create_model(config['base_model'], pretrained=True, num_classes=10)
model = vit_to_pit(base_model, (image_size,) * 2).to(device)

nas_names, nas_parameters = zip(*model.named_nas_parameters())
parameters = list(map(lambda x: x[1], filter(lambda x: x[0] not in nas_names, model.named_parameters())))
optimizer = optim.AdamW([{'params': parameters, 'lr': config['learning_rate'], 'weight_decay': config['weight_decay']},
                         {'params': nas_parameters, 'lr': config['nas_learning_rate'], 'weight_decay': 0}])
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=4, T_mult=1, eta_min=1e-8)
criterion = SoftTargetCrossEntropy()

def evaluate(model, data_loader):
    model.eval()
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
        return loss, correct, model.get_size_binarized()

def train(train_loader, model, optimizer, scheduler, criterion, size_lambda, frequency=10):
    model.train()
    mixup = Mixup(mixup_alpha=0.8, cutmix_alpha=1.0, label_smoothing=0.1, num_classes=model.n_classes)
    random_erasing = RandomErasing(probability=0.25, mode='pixel', device = device)
    scaler = torch.cuda.amp.GradScaler()
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

def checkpoint(model, optimizer, scheduler, name, epoch, config):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'config': config
    }, f"{name}_checkpoint_{epoch}.pt")

# initialize wandb
accuracy_stop = EarlyStopping(mode='max', patience=10)
size_stop = EarlyStopping(mode='min', patience=10)
run = wandb.init(project='pit-vit', config=config)
wandb.watch(model)

for epoch in range(config['epochs']):
    print(f"Epoch {epoch}")
    train(train_dl, model, optimizer, scheduler, criterion, size_lambda=config['size_lambda'], frequency=config['weight_update_frequency'])
    train_loss, train_accuracy, _, = evaluate(model, train_dl)
    print(f"TRAIN loss: {train_loss}, accuracy: {train_accuracy}")
    val_loss, val_accuracy, _ = evaluate(model, val_dl)
    print(f"VAL loss: {val_loss}, accuracy: {val_accuracy}")
    test_loss, test_accuracy, model_size = evaluate(model, test_dl)
    print(f"TEST loss: {test_loss}, accuracy: {test_accuracy}, model size: {model_size}")
    metrics = {'train_loss': train_loss, 'train_accuracy': train_accuracy,\
               'val_loss': val_loss, 'val_accuracy': val_accuracy, \
               'test_loss': test_loss, 'test_accuracy': test_accuracy, \
               'model_size': model_size}
    layers_params = dict()
    for name, param in model.named_modules():
        if isinstance(param, PITConv2d) or isinstance(param, PITLinear):
            layers_params[f"{name}"] = param.out_features_opt
    wandb.log(add_prefix("metrics/", metrics) | add_prefix("layers_masks/", layers_params))
    if epoch and epoch % config['checkpoint_frequency'] == 0:
        checkpoint(model, optimizer, scheduler, run.name, epoch, config)
    if accuracy_stop(val_accuracy) and size_stop(model_size):
        checkpoint(model, optimizer, scheduler, run.name, epoch, config)
        print("Early stopping!")
        break
wandb.finish()
