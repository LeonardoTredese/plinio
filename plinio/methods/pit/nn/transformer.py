from typing import Dict, Any, Optional, cast, Iterator, Tuple, Final
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx as fx
from plinio.graph.features_calculation import ConstFeaturesCalculator, FeaturesCalculator, ModAttrFeaturesCalculator
from .features_masker import PITFeaturesMasker
from .binarizer import PITBinarizer
from .module import PITModule
from .linear import PITLinear
from .conv2d import PITConv2d

class PITAttention(nn.Module, PITModule):
    def __init__(self,
                 hidden_dim: int,
                 n_heads: int,
                 binarization_threshold: float = 0.5,
                 qkv_bias: bool = False,
                 out_bias: bool = False,
                 fused_attention: bool = False,
                 ):
        super(PITAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // n_heads
        assert self.head_dim * n_heads == hidden_dim, "hidden_dim must be divisible by n_heads"
        self.scale_factor = self.head_dim ** -.5
        self.binarization_threshold = binarization_threshold
        self.n_heads = n_heads
        self.qk_features_masker = PITFeaturesMasker(self.hidden_dim)
        self.v_features_masker = PITFeaturesMasker(self.hidden_dim)
        self.output_features_masker = PITFeaturesMasker(self.hidden_dim)
        self.fused_attention = fused_attention
        
        self.q_proj = PITLinear(nn.Linear(hidden_dim, hidden_dim, bias=qkv_bias), \
                                self.qk_features_masker, binarization_threshold=binarization_threshold)
        self.k_proj = PITLinear(nn.Linear(hidden_dim, hidden_dim, bias=qkv_bias), \
                                self.qk_features_masker, binarization_threshold=binarization_threshold)
        self.v_proj = PITLinear(nn.Linear(hidden_dim, hidden_dim, bias=qkv_bias), \
                                self.v_features_masker, binarization_threshold=binarization_threshold)
        self.out_proj = PITLinear(nn.Linear(hidden_dim, hidden_dim, bias=out_bias), \
                                  self.output_features_masker, binarization_threshold=binarization_threshold)
        self.input_features_calculator = ConstFeaturesCalculator(hidden_dim)
        v_proj_calculator = ModAttrFeaturesCalculator(self.v_proj, 'out_features_opt', 'features_mask')
        self.out_proj.input_features_calculator = v_proj_calculator


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.reshape(*q.shape[:-1], self.n_heads, self.head_dim).transpose(-2, -3)
        k = k.reshape(*k.shape[:-1], self.n_heads, self.head_dim).transpose(-2, -3)
        v = v.reshape(*v.shape[:-1], self.n_heads, self.head_dim).transpose(-2, -3)
       
        if self.fused_attention:
            x = F.scaled_dot_product_attention(q, k, v) 
        else:
            q = q * self.scale_factor
            attention = q @ k.transpose(-2, -1)
            attention = torch.softmax(attention, dim=-1)
            x = attention @ v
        x = x.transpose(-2, -3).reshape(B, N, C)
        x = self.out_proj(x)
        return x

    @property
    def out_features_opt(self) -> int:
        return int(torch.sum(self.features_mask))

    @property
    def in_features_opt(self) -> int:
        with torch.no_grad():
            return int(torch.sum(self.input_features_calculator.features_mask))
    
    def __feature_mask(self, masker: PITFeaturesMasker) -> torch.Tensor:
        with torch.no_grad():
            theta_alpha = masker.theta
            return PITBinarizer.apply(theta_alpha, self.binarization_threshold)

    @property
    def features_mask(self) -> torch.Tensor:
        return self.__feature_mask(self.output_features_masker)

    @property
    def qk_features_mask(self) -> torch.Tensor:
        return self.__feature_mask(self.qk_features_masker)
    
    @property
    def v_features_mask(self) -> torch.Tensor:
        return self.__feature_mask(self.v_features_masker)

    def get_size(self) -> int:
        return self.q_proj.get_size() + self.k_proj.get_size() + self.v_proj.get_size() + self.out_proj.get_size()
    
    # TODO: implement
    def get_macs(self) -> int:
        return 0

    @property
    def input_features_calculator(self) -> FeaturesCalculator:
        return self._input_features_calculator
    
    @input_features_calculator.setter
    def input_features_calculator(self, input_features_calculator: FeaturesCalculator) -> None:
        self._input_features_calculator = input_features_calculator
        self.q_proj.input_features_calculator = input_features_calculator
        self.k_proj.input_features_calculator = input_features_calculator
        self.v_proj.input_features_calculator = input_features_calculator

    def summary(self) -> Dict[str, Any]:
        return { 'in_features': self.in_features_opt, 'out_features': self.out_features_opt, 'qk_features': int(torch.sum(self.qk_features_mask)), 'v_features': int(torch.sum(self.v_features_mask)) }

    def named_nas_parameters(self, prefix: str = '', recurse: bool = False) -> Iterator[Tuple[str, nn.Parameter]]:
        if prefix:
            prefix += '.'
        for name, param in self.qk_features_masker.named_parameters(prefix=prefix+'qk_features_masker', recurse=recurse):
            yield name, param
        for name, param in self.v_features_masker.named_parameters(prefix=prefix+'v_features_masker', recurse=recurse):
            yield name, param
        for name, param in self.output_features_masker.named_parameters(prefix=prefix+'output_features_masker', recurse=recurse):
            yield name, param

    def nas_parameters(self, recurse: bool = False) -> Iterator[nn.Parameter]:
        for _, param in self.named_nas_parameters(recurse=recurse):
            yield param

class PITMlp(nn.Module, PITModule):
    def __init__(self, d_model, scale, dropout, bias = False):
        super().__init__()
        self.d_model = d_model
        self.scale = scale
        hidden_dim = int(d_model * scale)
        self.mask_1 = PITFeaturesMasker(hidden_dim)
        self.mask_2 = PITFeaturesMasker(d_model)
        self.fc_1 = PITLinear(nn.Linear(d_model, hidden_dim, bias=bias), self.mask_1)
        self.fc_2 = PITLinear(nn.Linear(hidden_dim, d_model, bias=bias), self.mask_2)
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

    def named_nas_parameters(self, recurse: bool = False) -> Iterator[Tuple[str, nn.Parameter]]:
        for name, param in self.fc_1.named_nas_parameters(recurse=recurse):
            yield name, param
        for name, param in self.fc_2.named_nas_parameters(recurse=recurse):
            yield name, param

    def nas_parameters(self, recurse: bool = False) -> Iterator[nn.Parameter]:
        for _, param in self.named_nas_parameters(recurse=recurse):
            yield param

class PITBlock(nn.Module, PITModule):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            out_bias=False,
            proj_drop=0.,
            init_values=None,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            mlp_layer=PITMlp,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = PITAttention(
            dim,
            num_heads,
            qkv_bias=qkv_bias,
            out_bias=out_bias,
        )

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            dim,
            mlp_ratio,
            proj_drop,
            bias=True,
        )

        self.mlp.in_features_calculator = ModAttrFeaturesCalculator(self.attn, 'out_features_opt', 'features_mask')

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

    @property
    def out_features_opt(self):
        return self.mlp.out_features_opt

    @property
    def in_features_opt(self):
        return self.attn.in_features_opt

    @property
    def features_mask(self):
        return self.mlp.features_mask

    @property
    def input_features_calculator(self) -> FeaturesCalculator:
        return self.attn.input_features_calculator

    @input_features_calculator.setter
    def input_features_calculator(self, calc: FeaturesCalculator):
        self.attn.input_features_calculator = calc

    def get_size(self):
        return self.attn.get_size() + self.mlp.get_size()

    def get_macs(self):
        return self.attn.get_macs() + self.mlp.get_macs()

    def named_nas_parameters(self, recurse: bool = False) -> Iterator[Tuple[str, nn.Parameter]]:
        for name, param in self.attn.named_nas_parameters(recurse=recurse):
            yield name, param
        for name, param in self.mlp.named_nas_parameters(recurse=recurse):
            yield name, param

    def nas_parameters(self, recurse: bool = False) -> Iterator[nn.Parameter]:
        for _, param in self.named_nas_parameters(recurse=recurse):
            yield param

class PITPatchEmbedding(nn.Module, PITModule):
    def __init__(self, image_size, patch_size, d_model, bias = False):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.conv = PITConv2d(nn.Conv2d(3, d_model, patch_size, stride=patch_size, bias=bias), image_size[0], image_size[1], PITFeaturesMasker(d_model))

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

    def named_nas_parameters(self, recurse: bool = False) -> Iterator[Tuple[str, nn.Parameter]]:
        return self.conv.named_nas_parameters()
    
    def nas_parameters(self, recurse: bool = False) -> Iterator[nn.Parameter]:
        for _, param in self.named_nas_parameters(recurse=recurse):
            yield param

class PITVIT(nn.Module):
    def __init__(self, image_size, patch_size, n_layers, n_heads, d_model, ff_scale, dropout, n_classes, bias):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.ff_scale = ff_scale
        self.dropout = dropout
        self.n_classes = n_classes

        self.image_size = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        self.patch_embedding = PITPatchEmbedding(self.image_size, self.patch_size, d_model)
        self.num_patches = (self.image_size[0] // self.patch_size[0]) * (self.image_size[1] // self.patch_size[1])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches+1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.2)
        nn.init.trunc_normal_(self.pos_embedding, std=0.2)
        self.embed_dropout = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[PITBlock(d_model, n_heads, mlp_ratio=ff_scale, proj_drop=dropout, qkv_bias=bias, out_bias=bias) for _ in range(n_layers)])
        embed_features_calculator = ModAttrFeaturesCalculator(self.patch_embedding, 'out_features_opt', 'features_mask')
        for layer in self.blocks.children():
            layer.input_features_calculator = embed_features_calculator
        self.norm = nn.LayerNorm(d_model)
        self.head_dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, n_classes, bias = bias)
    
    def _pos_embed(self, x):
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        return x + self.pos_embedding

    def forward_features(self, x):
        x = self.patch_embedding(x)
        x = self._pos_embed(x)
        x = self.embed_dropout(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = x[:, 0]
        x = self.head_dropout(x)
        x = self.head(x)
        return x

    def get_size(self):
        size = self.patch_embedding.get_size() 
        for layer in self.blocks.children():
            size += layer.get_size()
        return size
