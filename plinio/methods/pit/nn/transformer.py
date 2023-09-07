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
import timm

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
        self.qk_features_masker = PITFeaturesMasker(self.head_dim)
        self.v_features_masker = PITFeaturesMasker(self.head_dim)
        self.output_features_masker = PITFeaturesMasker(self.hidden_dim)
        self.heads_features_masker = PITFeaturesMasker(n_heads)
        self.fused_attention = fused_attention
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=qkv_bias) 
        self.out_proj = PITLinear(nn.Linear(hidden_dim, hidden_dim, bias=out_bias), \
                                  self.output_features_masker, binarization_threshold=binarization_threshold)
        self.input_features_calculator = ConstFeaturesCalculator(hidden_dim)
        v_proj_calculator = ModAttrFeaturesCalculator(self.v_proj, 'out_features_opt', 'features_mask')
        self.out_proj.input_features_calculator = v_proj_calculator
        self.register_buffer('heads_eff', torch.tensor(self.n_heads, dtype=torch.float32))

    @staticmethod  
    def from_timm(attention: timm.models.vision_transformer.Attention):
        hidden_dim = attention.num_heads * attention.head_dim
        qkv_bias = attention.qkv.bias is not None
        out_bias = attention.proj.bias is not None
        fused_attn = attention.fused_attn
        pit = PITAttention(hidden_dim, attention.num_heads, qkv_bias=qkv_bias, out_bias=out_bias, fused_attention=fused_attn)
        pit.q_proj.weight.data, pit.k_proj.weight.data, pit.v_proj.weight.data= attention.qkv.weight.data.chunk(3)
        if qkv_bias:
            pit.q_proj.bias.data, pit.k_proj.bias.data, pit.v_proj.bias.data= attention.qkv.bias.data.chunk(3)
        pit.out_proj.weight.data = attention.proj.weight.data
        if out_bias:
            pit.out_proj.bias.data = attention.proj.bias.data
        return pit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        heads_masks = self._feature_mask(self.heads_features_masker, discrete=True)
        qk_head_mask = self._feature_mask(self.qk_features_masker, discrete=True)
        v_head_mask = self._feature_mask(self.v_features_masker, discrete=True) 
        qk_mask = (heads_masks.unsqueeze(1) * qk_head_mask.unsqueeze(0)).flatten()
        v_mask = (heads_masks.unsqueeze(1) * v_head_mask.unsqueeze(0)).flatten()

        B, N, C = x.shape
        q_pruned = torch.mul(self.q_proj.weight, qk_mask.unsqueeze(1))
        q = nn.functional.linear(x, q_pruned, self.q_proj.bias)
        k_pruned = torch.mul(self.k_proj.weight, qk_mask.unsqueeze(1))
        k = nn.functional.linear(x, k_pruned, self.k_proj.bias)
        v_pruned = torch.mul(self.v_proj.weight, v_mask.unsqueeze(1))
        v = nn.functional.linear(x, v_pruned, self.v_proj.bias)

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
   
    def _feature_mask(self, masker: PITFeaturesMasker, discrete: bool) -> torch.Tensor:
        theta_alpha = masker.theta
        if discrete:
            theta_alpha = PITBinarizer.apply(theta_alpha, self.binarization_threshold)
        return cast(torch.Tensor, theta_alpha)

    def __feature_mask(self, masker: PITFeaturesMasker) -> torch.Tensor:
        with torch.no_grad():
            theta_alpha = masker.theta
            return PITBinarizer.apply(theta_alpha, self.binarization_threshold)
    
    @property
    def heads_mask(self) -> torch.Tensor:
        return self.__feature_mask(self.heads_features_masker)

    @property
    def features_mask(self) -> torch.Tensor:
        return self.__feature_mask(self.output_features_masker)

    @property
    def qk_features_mask(self) -> torch.Tensor:
        return self.__feature_mask(self.qk_features_masker)
    
    @property
    def v_features_mask(self) -> torch.Tensor:
        return self.__feature_mask(self.v_features_masker)

    def get_size(self) -> float:
        in_features = self.input_features_calculator.features
        heads_eff = torch.sum(self.heads_features_masker.theta)
        qk_eff = torch.sum(self.qk_features_masker.theta)
        v_eff = torch.sum(self.v_features_masker.theta)
        return in_features * heads_eff * (2 * qk_eff + v_eff) + heads_eff * v_eff * self.out_proj.out_features_eff

    def get_size_binarized(self) -> int:
        in_features = torch.sum(self.input_features_calculator.features_mask)
        heads = torch.sum(self.__feature_mask(self.heads_features_masker))
        qk = torch.sum(self.__feature_mask(self.qk_features_masker))
        v = torch.sum(self.__feature_mask(self.v_features_masker))
        return in_features * heads * (2 * qk + v) + heads * v * self.out_features_opt
    
    # TODO: implement
    def get_macs(self) -> int:
        return 0

    @property
    def input_features_calculator(self) -> FeaturesCalculator:
        return self._input_features_calculator
    
    @input_features_calculator.setter
    def input_features_calculator(self, input_features_calculator: FeaturesCalculator) -> None:
        self._input_features_calculator = input_features_calculator

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
        for name, param in self.heads_features_masker.named_parameters(prefix=prefix+'heads_features_masker', recurse=recurse):
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
        self.fc_1 = PITLinear(nn.Linear(d_model, hidden_dim, bias=bias), PITFeaturesMasker(hidden_dim))
        self.fc_2 = PITLinear(nn.Linear(hidden_dim, d_model, bias=bias), PITFeaturesMasker(d_model))
        fc1_features_calculator = ModAttrFeaturesCalculator(self.fc_1, 'out_features_opt', 'features_mask')
        self.fc_2.input_features_calculator = fc1_features_calculator
        self.activation = nn.GELU()
        self.drop_1 = nn.Dropout(dropout)
        self.drop_2 = nn.Dropout(dropout)

    @staticmethod
    def from_timm(mlp: timm.layers.Mlp):
        in_dim = mlp.fc1.weight.shape[1]
        scale = mlp.fc1.weight.shape[0] // in_dim
        has_bias = mlp.fc1.bias is not None
        pit_mlp = PITMlp(in_dim, scale, mlp.drop1.p, bias=has_bias)
        pit_mlp.fc_1.weight.data = mlp.fc1.weight.data
        pit_mlp.fc_2.weight.data = mlp.fc2.weight.data
        if has_bias:
            pit_mlp.fc_1.bias.data = mlp.fc1.bias.data
            pit_mlp.fc_2.bias.data = mlp.fc2.bias.data
        return pit_mlp

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

    def get_size(self) -> float:
        return self.fc_1.get_size() + self.fc_2.get_size()

    def get_size_binarized(self) -> int:
        return self.fc_1.get_size_binarized() + self.fc_2.get_size_binarized()

    def get_macs(self):
        return 0

    def named_nas_parameters(self, recurse: bool = False) -> Iterator[Tuple[str, nn.Parameter]]:
        for name, param in self.fc_1.named_nas_parameters(recurse=recurse):
            yield f"fc_1.{name}", param
        for name, param in self.fc_2.named_nas_parameters(recurse=recurse):
            yield f"fc_2.{name}", param

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
    @staticmethod
    def from_timm(block: timm.models.vision_transformer.Block):
        dim = block.attn.num_heads * block.attn.head_dim
        pit_block = PITBlock(dim, block.attn.num_heads)
        pit_block.attn = PITAttention.from_timm(block.attn)
        pit_block.mlp = PITMlp.from_timm(block.mlp)
        pit_block.norm1.load_state_dict(block.norm1.state_dict())
        pit_block.norm2.load_state_dict(block.norm2.state_dict())
        pit_block.norm1.eps = block.norm1.eps
        pit_block.norm2.eps = block.norm2.eps
        return pit_block

    def forward(self, x):
        attn_out = self.attn.out_proj
        theta_alpha = attn_out.out_features_masker.theta
        attn_mask = PITBinarizer.apply(theta_alpha, attn_out._binarization_threshold)
        mlp_out = self.mlp.fc_2
        theta_alpha = mlp_out.out_features_masker.theta
        mlp_mask = PITBinarizer.apply(theta_alpha, mlp_out._binarization_threshold)
        x = torch.mul(x, attn_mask) + self.attn(self.norm1(x))
        x = torch.mul(x, mlp_mask) + self.mlp(self.norm2(x))
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
        self.mlp.input_features_calculator = calc

    def get_size(self) -> float:
        return self.attn.get_size() + self.mlp.get_size()

    def get_size_binarized(self) -> int:
        return self.attn.get_size_binarized() + self.mlp.get_size_binarized()

    def get_macs(self):
        return self.attn.get_macs() + self.mlp.get_macs()

    def named_nas_parameters(self, recurse: bool = False) -> Iterator[Tuple[str, nn.Parameter]]:
        for name, param in self.attn.named_nas_parameters(recurse=recurse):
            yield f"attn.{name}", param
        for name, param in self.mlp.named_nas_parameters(recurse=recurse):
            yield f"mlp.{name}", param

    def nas_parameters(self, recurse: bool = False) -> Iterator[nn.Parameter]:
        for _, param in self.named_nas_parameters(recurse=recurse):
            yield param

class PITPatchEmbedding(nn.Module, PITModule):
    def __init__(self, image_size, patch_size, d_model, bias = False):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.conv = PITConv2d(nn.Conv2d(3, d_model, patch_size, stride=patch_size, bias=bias), image_size[0], image_size[1], PITFeaturesMasker(d_model))

    @staticmethod
    def from_timm(patch_embed: timm.layers.PatchEmbed, image_size):
        patch_size = patch_embed.proj.kernel_size
        hidden_dim = patch_embed.proj.weight.shape[0]
        has_bias = patch_embed.proj.bias is not None
        pit_embed = PITPatchEmbedding(image_size, patch_size, hidden_dim, bias = has_bias)
        pit_embed.conv.weight.data = patch_embed.proj.weight.data
        if has_bias:
            pit_embed.conv.bias.data = patch_embed.proj.bias.data
        return pit_embed

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
    
    def get_size(self) -> float:
        return self.conv.out_features_eff * 3 * self.patch_size[0] * self.patch_size[1]

    def get_size_binarized(self) -> int:
        return self.conv.out_features_opt * 3 * self.patch_size[0] * self.patch_size[1]

    def get_macs(self):
        return 0

    def named_nas_parameters(self, recurse: bool = False) -> Iterator[Tuple[str, nn.Parameter]]:
        for name, param in self.conv.named_nas_parameters():
            yield f"conv.{name}", param
    
    def nas_parameters(self, recurse: bool = False) -> Iterator[nn.Parameter]:
        for _, param in self.named_nas_parameters(recurse=recurse):
            yield param

class PITVIT(nn.Module, PITModule):
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
        self.norm = nn.LayerNorm(d_model)
        self.head_dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, n_classes, bias = bias)
        self.cascade_wiring()

    @staticmethod
    def from_timm(vit: timm.models.vision_transformer.VisionTransformer, image_size):
        n_layers = len(vit.blocks)
        sample_block = vit.blocks[0]
        n_heads = sample_block.attn.num_heads
        d_model = n_heads * sample_block.attn.head_dim
        ff_scale = sample_block.mlp.fc1.weight.shape[0] // d_model
        dropout = sample_block.mlp.drop1.p
        bias = sample_block.attn.qkv.bias is not None
        n_classes = vit.head.out_features
        patch_size = vit.patch_embed.proj.kernel_size
        pit_vit = PITVIT(image_size, patch_size, n_layers, n_heads, d_model, ff_scale, dropout, n_classes, bias)
        pit_vit.cls_token.data = vit.cls_token.data
        pit_vit.pos_embedding.data = vit.pos_embed.data
        pit_vit.patch_embedding = PITPatchEmbedding.from_timm(vit.patch_embed, image_size)
        embed_features_calculator = ModAttrFeaturesCalculator(pit_vit.patch_embedding, 'out_features_opt', 'features_mask')
        for i, block in enumerate(vit.blocks.children()):
            pit_vit.blocks[i] = PITBlock.from_timm(block)
        pit_vit.norm.load_state_dict(vit.norm.state_dict())
        pit_vit.norm.eps = vit.norm.eps
        pit_vit.head.weight.data = vit.head.weight.data
        if vit.head.bias is not None:
            pit_vit.head.bias.data = vit.head.bias.data
        return pit_vit

    def cascade_wiring(self):
        prev_features_calculator = ModAttrFeaturesCalculator(self.patch_embedding, 'out_features_opt', 'features_mask')
        for block in self.blocks.children():
            block.input_features_calculator = prev_features_calculator
            in_features_calculator = ModAttrFeaturesCalculator(block, 'out_features_opt', 'features_mask')

    def hidden_dim_wiring(self):
        hidden_dim_masker = PITFeaturesMasker(self.d_model)
        self.patch_embedding.conv.out_features_masker = hidden_dim_masker
        in_features_calculator = ModAttrFeaturesCalculator(self.patch_embedding, 'out_features_opt', 'features_mask')
        for block in self.blocks.children():
            block.input_features_calculator = in_features_calculator
            block.attn.out_proj.out_features_masker = hidden_dim_masker
            block.mlp.fc_2.out_features_masker = hidden_dim_masker

    def shared_mask_wiring(self):
        head_dim = self.d_model // self.n_heads
        hidden_dim_masker = PITFeaturesMasker(self.d_model)
        qk_masker = PITFeaturesMasker(head_dim)
        v_masker = PITFeaturesMasker(head_dim)
        heads_features_masker = PITFeaturesMasker(self.n_heads)
        fc1_masker = PITFeaturesMasker(self.d_model * self.ff_scale)
        self.patch_embedding.conv.out_features_masker = hidden_dim_masker
        in_features_calculator = ModAttrFeaturesCalculator(self.patch_embedding, 'out_features_opt', 'features_mask')
        for block in self.blocks.children():
            block.input_features_calculator = in_features_calculator
            block.attn.out_proj.out_features_masker = hidden_dim_masker
            block.attn.qk_features_masker = qk_masker
            block.attn.v_features_masker = v_masker
            block.attn.heads_features_masker = heads_features_masker
            block.mlp.fc_1.out_features_masker = fc1_masker
            block.mlp.fc_2.out_features_masker = hidden_dim_masker

    def _pos_embed(self, x):
        embed_conv = self.patch_embedding.conv
        theta_alpha = embed_conv.out_features_masker.theta
        embed_mask = PITBinarizer.apply(theta_alpha, embed_conv._binarization_threshold)
        masked_cls_token = torch.mul(self.cls_token, embed_mask)
        x = torch.cat((masked_cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        return x + torch.mul(self.pos_embedding, embed_mask)

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

    def get_size(self) -> float:
        size = self.patch_embedding.get_size() + self.head.weight.numel()
        for layer in range(self.n_layers):
            block = self.blocks[layer]
            size += block.get_size()
        return size

    def get_size_binarized(self) -> int:
        size = self.patch_embedding.get_size_binarized() + self.head.weight.numel()
        for layer in range(self.n_layers):
            block = self.blocks[layer]
            size += block.get_size_binarized()
        return size

    @property
    def in_features_opt(self):
        return 3

    def get_macs(self):
        return 0

    @property
    def out_features_opt(self):
        return self.n_classes

    @property
    def features_mask(self):
        return torch.ones_like(self.head.weight)

    @property
    def input_features_calculator(self) -> FeaturesCalculator:
        return self.conv.input_features_calculator

    @input_features_calculator.setter
    def input_features_calculator(self, calc: FeaturesCalculator):
        self.conv.input_features_calculator = calc

    def named_nas_parameters(self, recurse: bool = False) -> Iterator[Tuple[str, nn.Parameter]]:
        for name, param in self.patch_embedding.named_nas_parameters(recurse=recurse):
            yield f"patch_embedding.{name}", param
        for layer in range(self.n_layers):
            block = self.blocks[layer]
            for name, param in block.named_nas_parameters(recurse=recurse):
                yield f"blocks.{layer}.{name}", param
    
    def nas_parameters(self, recurse: bool = False) -> Iterator[nn.Parameter]:
        for _, param in self.named_nas_parameters(recurse=recurse):
            yield param
