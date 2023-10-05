import torch
import torch.nn as nn
import plinio.methods.pit.nn as pnn
from plinio.methods.pit.nn import PITVIT
from plinio.methods.pit.nn.vit import VIT
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

device = 'cuda:3'
dev_id = int(device[-1])

# relocate memory to specified memory
def map_location(storage, dev):
    return storage if dev != 'cpu'else storage.cuda(dev_id)
    
# ensure deterministic behavior
seed = seed_all(seed=42)

# load saved model information
file_path = '/home/tredese/experiments/generous-pine-228_checkpoint_372.pt'
checkpoint = torch.load(file_path, map_location = map_location)
config = checkpoint['config']
config['learning_rate'] = 1e-4

image_size = config['image_size']
patch_size = config['patch_size']
datasets = icl.get_data(dataset=config['dataset'], download=True, image_size=(image_size, image_size), data_dir ='~/.cache')
dataloaders = icl.build_dataloaders(datasets, batch_size=config['batch_size'], num_workers=os.cpu_count(), seed=seed)
train_dl, val_dl, test_dl = dataloaders

dataset_classes = {'cifar10': 10, 'tiny-imagenet': 200}
base_model = timm.create_model(config['base_model'], pretrained=True, num_classes=dataset_classes[config['dataset']])
pit = PITVIT.from_timm(base_model, (image_size,) * 2)
pit.load_state_dict(checkpoint['model_state_dict'])
pit.set_in_features_calculators()
model = VIT.from_pit(pit)
model = model.to(device)

optimizer = optim.AdamW(model.parameters(), lr = config['learning_rate'], weight_decay= config['weight_decay'])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=0)
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
        return loss, correct 

def train(train_loader, model, optimizer, scheduler, criterion, frequency=10):
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
            loss = criterion(output, target)
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
run = wandb.init(project='vit-finetuning', config=config)
wandb.watch(model)

test_loss, test_accuracy = evaluate(model, test_dl)
model_size = sum(p.numel() for p in model.parameters())
print(f"BEFORE TRAINING: TEST loss: {test_loss}, accuracy: {test_accuracy}, size {model_size} ")

for epoch in range(config['epochs']):
    print(f"Epoch {epoch}")
    train(train_dl, model, optimizer, scheduler, criterion, frequency=config['weight_update_frequency'])
    val_loss, val_accuracy  = evaluate(model, val_dl)
    print(f"VAL loss: {val_loss}, accuracy: {val_accuracy}")
    test_loss, test_accuracy = evaluate(model, test_dl)
    print(f"TEST loss: {test_loss}, accuracy: {test_accuracy}")
    metrics = { 'val_loss': val_loss, 'val_accuracy': val_accuracy, \
               'test_loss': test_loss, 'test_accuracy': test_accuracy}
    wandb.log(add_prefix("metrics/", metrics))
    if epoch and epoch % config['checkpoint_frequency'] == 0:
        checkpoint(model, optimizer, scheduler, run.name, epoch, config)
    if accuracy_stop(val_accuracy):
        checkpoint(model, optimizer, scheduler, run.name, epoch, config)
        print("Early stopping!")
        break
wandb.finish()
