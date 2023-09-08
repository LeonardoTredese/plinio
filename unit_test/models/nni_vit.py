import torch
import torch.nn as nn
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
import nni
from nni.compression.pytorch.pruning import L2NormPruner
from nni.compression.pytorch.speedup import ModelSpeedup
import functools

# train the model on cifar10
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ensure deterministic behavior
seed = seed_all(seed=42)

fine_tuned_model_name = 'vit_cifar10.pth'

# configuration
config = {
    'base_model': 'vit_tiny_patch16_384',
    'dataset': 'cifar10',
    'batch_size': 128,
    'learning_rate': 1e-3,
    'nas_learning_rate': 1e-2,
    'weight_decay': 5e-2,
    'epochs': 250,
    'size_lambda': 1e-7,
    'weight_update_frequency': 10,
    'image_size': 384,
    'patch_size': 16,
    'checkpoint_frequency': 10,
}

# cifar10 stardard meand and deviation
image_size = config['image_size']
patch_size = config['patch_size']
datasets = icl.get_data(dataset=config['dataset'], download=True, image_size=(image_size, image_size))
dataloaders = icl.build_dataloaders(datasets, batch_size=config['batch_size'], num_workers=os.cpu_count(), seed=seed)
train_dl, val_dl, test_dl = dataloaders

model = timm.create_model(config['base_model'], pretrained=True, num_classes=10)
if fine_tuned_model_name:
    model.load_state_dict(torch.load(fine_tuned_model_name))
else:
    print("not using fine tuned model")
model = model.to(device)

traced_optimizer = nni.trace(optim.AdamW)(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
traced_scheduler = nni.trace(optim.lr_scheduler.CosineAnnealingLR)(traced_optimizer, T_max=config['epochs'])
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

def train_iteration(model, optimizer, criterion, scheduler=None, train_loader = None, frequency=None):
    model.train()
    mixup = Mixup(mixup_alpha=0.8, cutmix_alpha=1.0, label_smoothing=0.1, num_classes=model.num_classes)
    random_erasing = RandomErasing(probability=0.25, mode='pixel', device = device)
    scaler = torch.cuda.amp.GradScaler()
    for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        data, target = data.to(device), target.to(device)
        data, target = mixup(data, target)
        data = random_erasing(data)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        # gradient accumulation
        if (batch_idx + 1) % frequency == 0 or batch_idx == len(train_loader) - 1:
            optimizer.step()
            optimizer.zero_grad()
    scheduler.step()

def train(model, optimizer, criterion, scheduler=None, train_loader=None, frequency=None, max_epoch=None):
    for epoch in range(max_epoch):
        train_iteration(model, optimizer, criterion, scheduler, train_loader, frequency)

prune_config = [{
    'op_types': ['Linear', 'Conv2d'],
    'op_partial_names': ['blocks', 'patch_embed'],
    'sparsity': 0.75,
}, 
{
    'exclude': True,
    'op_names': ['head'],
}
]

dummy_input = torch.randn(1, 3, image_size, image_size).to(device)

pruner = L2NormPruner(model, prune_config, mode = 'dependency_aware', dummy_input = dummy_input)

model_size = model.head.weight.numel()
_, masks = pruner.compress()

for m_name, mask in masks.items():
    tensor = mask['weight']
    model_size += tensor.sum().item()
pruner._unwrap_model()
ModelSpeedup(model, dummy_input, masks).speedup_model()
print(f"model size: {model_size}")
loss, accuracy = evaluate(model, test_dl)
print(f"loss: {loss}, accuracy: {accuracy}")


# store the pruned model
# torch.save(model.state_dict(), 'pruned_model.pth')
# 
# optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
# 
# for train_iter in range(config['epochs']):
#     print (f"train iteration: {train_iter}")
#     train(model, optimizer, criterion, scheduler, train_dl, config['weight_update_frequency'], 1)
#     loss, accuracy = evaluate(model, test_dl)
#     print(f"loss: {loss}, accuracy: {accuracy}")
#     # store prunedfine tuned model
#     torch.save(model.state_dict(), 'pruned_finetuned_model.pth')
