import torch
import torch.nn as nn
import plinio.methods.pit.nn as pnn
from plinio.methods.pit.nn.features_masker import PITFeaturesMasker
from plinio.graph.features_calculation import ModAttrFeaturesCalculator, FeaturesCalculator

class DumbModel(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_size = 3 * patch_size ** 2
        self.patcher = PatchEmbedding([image_size] * 2, [patch_size] * 2, hidden_size)
        self.n_classes = 10
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        torch.nn.init.xavier_normal_(self.cls_token)
        qk_masker = PITFeaturesMasker(hidden_size)
        v_masker = PITFeaturesMasker(hidden_size)
        out_masker = PITFeaturesMasker(hidden_size)
        self.encoder = pnn.PITMHSA(hidden_size, 8, qk_masker, v_masker, out_masker)
        self.head = nn.Linear(hidden_size, self.n_classes)

    def forward(self, x):
        x = self.patcher(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.encoder(x)
        x = self.head(x[:, 0])
        return x

    def get_size(self):
        return torch.mul(*self.head.weight.shape) + self.patcher.get_size()


class FeedForward(nn.Module, pnn.PITModule):
    def __init__(self, d_model, scale, dropout):
        super().__init__()
        self.d_model = d_model
        self.scale = scale
        self.fc_1 = pnn.PITLinear(nn.Linear(d_model, d_model * scale, bias=False), PITFeaturesMasker(d_model * scale))
        self.fc_2 = pnn.PITLinear(nn.Linear(d_model * scale, d_model, bias=False), PITFeaturesMasker(d_model))
        fc1_features_calculator = ModAttrFeaturesCalculator(self.fc_1, 'out_features_opt', 'features_mask')
        self.fc_2.input_features_calculator = fc1_features_calculator
        self.activation = nn.GELU()
        self.drop_1 = nn.Dropout(dropout)
        self.drop_2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc_1(x)
        x = self.activation(x)
        x = self.drop_1(x)
        x = self.fc_2(x)
        x = self.drop_2(x)
        return x
    
    @property
    def out_features_opt(self):
        return self.fc_2.out_features_opt

    @property
    def in_features_opt(self):
        return self.fc_1.in_features_opt

    @property
    def features_mask(self):
        return self.fc_2.features_mask

    @property
    def input_features_calculator(self) -> FeaturesCalculator:
        return self.fc_1.input_features_calculator

    @input_features_calculator.setter
    def input_features_calculator(self, calc: FeaturesCalculator):
        self.fc_1.input_features_calculator = calc

    def get_size(self):
        return self.fc_1.get_size() + self.fc_2.get_size()

    def get_macs(self):
        return 0

class EncoderLayer(nn.Module, pnn.PITModule):
    def __init__(self, n_heads, d_model, ff_scale, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.ff_scale = ff_scale
        self.dropout = dropout
        self.norm_1 = nn.LayerNorm(d_model)
        self.self_attention =  pnn.PITMHSA(d_model, n_heads, PITFeaturesMasker(d_model), PITFeaturesMasker(d_model), PITFeaturesMasker(d_model))
        self.attn_drop = nn.Dropout(dropout)
        self.norm_2 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, ff_scale, dropout)

    def forward(self, x):
        x = self.attn_drop(self.self_attention(self.norm_1(x))) + x
        x = self.feed_forward(self.norm_2(x)) + x
        return x
    
    @property
    def in_features_opt(self):
        return self.self_attention.in_features_opt

    @property
    def out_features_opt(self):
        return self.feed_forward.out_features_opt

    @property
    def features_mask(self):
        return self.feed_forward.features_mask

    @property
    def input_features_calculator(self) -> FeaturesCalculator:
        return self.self_attention.input_features_calculator

    @input_features_calculator.setter
    def input_features_calculator(self, calc: FeaturesCalculator):
        self.self_attention.input_features_calculator = calc
        self.feed_forward.input_features_calculator = calc
        
    def get_size(self):
        return self.self_attention.get_size() + self.feed_forward.get_size()

    def get_macs(self):
        return 0

class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, d_model):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.conv = pnn.PITConv2d(nn.Conv2d(3, d_model, patch_size, stride=patch_size, bias=False), image_size[0], image_size[1], PITFeaturesMasker(d_model))

    def forward(self, x):
        return self.conv(x).flatten(2).transpose(1, 2)

    @property
    def in_features_opt(self):
        return 3

    @property
    def out_features_opt(self):
        return self.conv.out_features_opt

    @property
    def features_mask(self):
        return self.conv.features_mask

    @property
    def input_features_calculator(self) -> FeaturesCalculator:
        return self.conv.input_features_calculator

    @input_features_calculator.setter
    def input_features_calculator(self, calc: FeaturesCalculator):
        self.conv.input_features_calculator = calc
    
    def get_size(self):
        return self.in_features_opt * self.out_features_opt

    def get_macs(self):
        return 0

class VisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, n_layers, n_heads, d_model, ff_scale, dropout, n_classes):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.ff_scale = ff_scale
        self.dropout = dropout
        self.n_classes = n_classes

        self.image_size = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        self.patch_embedding = PatchEmbedding(self.image_size, self.patch_size, d_model)
        self.num_patches = (self.image_size[0] // self.patch_size[0]) * (self.image_size[1] // self.patch_size[1])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches+1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.2)
        nn.init.trunc_normal_(self.pos_embedding, std=0.2)
        self.embed_dropout = nn.Dropout(dropout)
        self.encoder = nn.ModuleList([EncoderLayer(n_heads, d_model, ff_scale, dropout) for _ in range(n_layers)])
        embed_features_calculator = ModAttrFeaturesCalculator(self.patch_embedding, 'out_features_opt', 'features_mask')
        for layer in self.encoder:
            layer.input_features_calculator = embed_features_calculator
        self.norm = nn.LayerNorm(d_model)
        self.head_dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, n_classes, bias = False)

    def forward_features(self, x):
        x = self.patch_embedding(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding
        x = self.embed_dropout(x)
        for layer in self.encoder:
            x = layer(x)
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head_dropout(x)
        x = self.head(x)
        return x

    def get_size(self):
        size = self.patch_embedding.get_size() 
        for layer in self.encoder:
            size += layer.get_size()
        return size

# train the model on cifar10
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
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
image_size = 32
patch_size = 4
train_transform = torchvision.transforms.Compose([transforms.RandAugment(2, 9), transforms.Resize(image_size), transforms.ToTensor(), normalize])
test_transform = torchvision.transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(), normalize])
train_ds = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
test_ds = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

batch_size = 128
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)

model = VisionTransformer(image_size, patch_size, n_layers=4, n_heads=8, d_model=image_size * 3, ff_scale=4, dropout=0.1, n_classes=10).to(device)
model = DumbModel().to(device)
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
    random_erasing = RandomErasing(probability=0.25, mode='pixel')
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

