import torch
import torch.nn as nn
import torch.nn.functional as F
import plinio.methods.pit.nn as pnn

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, d_model, bias = False):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.conv = nn.Conv2d(3, d_model, patch_size, stride=patch_size, bias=bias)
    
    def forward(self, x):
        return self.conv(x).flatten(2).transpose(1, 2)

    @staticmethod
    def from_pit(patch_embed: pnn.PITPatchEmbedding):
        patch_size = patch_embed.conv.weight.shape[-2:]
        hidden_dim = patch_embed.out_features_opt
        has_bias = patch_embed.conv.bias is not None
        embed = PatchEmbedding(patch_size, hidden_dim, bias = has_bias)
        mask = patch_embed.features_mask.bool()
        embed.conv.weight.data = patch_embed.conv.weight.data[mask]
        if has_bias:
            embed.conv.bias.data = patch_embed.conv.bias.data[mask]
        return embed

class Attention(nn.Module):
    def __init__(self,
                 qk_dim: int,
                 v_dim: int,
                 hidden_dim: int,
                 n_heads: int,
                 binarization_threshold: float = 0.5,
                 qkv_bias: bool = False,
                 out_bias: bool = False,
                 fused_attention: bool = False,
                 ):
        super(Attention, self).__init__()
        self.scale_factor = qk_dim ** -.5
        self.binarization_threshold = binarization_threshold
        self.n_heads = n_heads
        self.fused_attention = fused_attention
        self.qk_dim = qk_dim
        self.v_dim = v_dim
        self.q_proj = nn.Linear(hidden_dim, qk_dim * n_heads, bias=qkv_bias)
        self.k_proj = nn.Linear(hidden_dim, qk_dim * n_heads, bias=qkv_bias)
        self.v_proj = nn.Linear(hidden_dim, v_dim * n_heads, bias=qkv_bias) 
        self.out_proj = nn.Linear(v_dim, hidden_dim, bias=out_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape(*q.shape[:-1], self.n_heads, self.qk_dim).transpose(-2, -3)
        k = k.reshape(*k.shape[:-1], self.n_heads, self.qk_dim).transpose(-2, -3)
        v = v.reshape(*v.shape[:-1], self.n_heads, self.v_dim).transpose(-2, -3)

        if self.fused_attention:
            x = F.scaled_dot_product_attention(q, k, v) 
        else:
            q = q * self.scale_factor
            attention = q @ k.transpose(-2, -1)
            attention = torch.softmax(attention, dim=-1)
            x = attention @ v
        x = x.transpose(-2, -3).reshape(B, N, self.v_dim * self.n_heads)
        x = self.out_proj(x)
        return x

    @staticmethod  
    def from_pit(attention: pnn.PITAttention):
        qkv_bias = attention.q_proj.bias is not None
        out_bias = attention.out_proj.bias is not None
        fused_attn = attention.fused_attention
        qk_dim = int(attention.qk_features_mask.sum())
        v_dim = int(attention.v_features_mask.sum())
        n_heads = int(attention.heads_features_mask.sum())
        hidden_dim = int(attention.input_features_calculator.features_mask.sum())
        attn = Attention(qk_dim, v_dim, hidden_dim, n_heads, binarization_threshold=attention.binarization_threshold, qkv_bias=qkv_bias, out_bias=out_bias, fused_attention=True)
        in_mask = attention.input_features_calculator.features_mask.bool()
        head_mask = attention.heads_features_mask.int().unsqueeze(1)
        qk_mask = (attention.qk_features_mask.int().unsqueeze(0) * head_mask).flatten().bool()
        v_mask = (attention.v_features_mask.int().unsqueeze(0) * head_mask).flatten().bool()
        out_mask = attention.features_mask.bool()

        attn.q_proj.weight.data = mask_weight(attention.q_proj.weight.data, in_mask, qk_mask)
        attn.k_proj.weight.data = mask_weight(attention.k_proj.weight.data, in_mask, qk_mask)
        attn.v_proj.weight.data = mask_weight(attention.v_proj.weight.data, in_mask, v_mask)
        attn.out_proj.weight.data = mask_weight(attention.out_proj.weight.data, v_mask, out_mask)
        if qkv_bias:
            attn.q_proj.bias.data = attention.q_proj.bias.data[qk_mask]
            attn.k_proj.bias.data = attention.k_proj.bias.data[qk_mask]
            attn.v_proj.bias.data = attention.v_proj.bias.data[v_mask]
        if out_bias:
            attn.out_proj.bias.data = attention.out_proj.bias.data[out_mask]
        return attn

class Mlp(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout, bias = False):
        super(Mlp, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.fc_1 = nn.Linear(in_dim, hidden_dim, bias=bias)
        self.fc_2 = nn.Linear(hidden_dim, out_dim, bias=bias)
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

    @staticmethod
    def from_pit(mlp: pnn.PITMlp):
        has_bias = mlp.fc_1.bias is not None
        in_dim = int(mlp.input_features_calculator.features)
        hidden = mlp.fc_1.out_features_opt
        out_dim = mlp.out_features_opt
        p = mlp.drop_1.p
        new_mlp = Mlp(in_dim, hidden, out_dim, p, bias = has_bias)
        in_mask = mlp.input_features_calculator.features_mask.bool()
        hid_mask = mlp.fc_1.features_mask.bool()
        out_mask = mlp.features_mask.bool()
        new_mlp.fc_1.weight.data = mask_weight(mlp.fc_1.weight.data, in_mask, hid_mask)
        new_mlp.fc_2.weight.data = mask_weight(mlp.fc_2.weight.data, hid_mask, out_mask)
        new_mlp.drop_1.p = p
        new_mlp.drop_2.p = p
        new_mlp.activation = mlp.activation
        if has_bias:
            new_mlp.fc_1.bias.data = mlp.fc_1.bias[hid_mask]
            new_mlp.fc_2.bias.data = mlp.fc_2.bias[out_mask]
        return new_mlp

class Block(nn.Module):
    def __init__(
            self,
            dim,
            norm_layer=nn.LayerNorm,
    ):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = None 
        self.norm2 = norm_layer(dim)
        self.mlp = None

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))

    @staticmethod
    def from_pit(block: pnn.PITBlock):
        dim = block.out_features_opt
        new_block = Block(dim)
        new_block.attn = Attention.from_pit(block.attn)
        new_block.mlp = Mlp.from_pit(block.mlp)
        mask = block.features_mask.bool()
        new_block.norm1.weight.data = block.norm1.weight.data[mask]
        new_block.norm2.weight.data = block.norm2.weight.data[mask]
        new_block.norm1.bias.data = block.norm1.bias.data[mask]
        new_block.norm2.bias.data = block.norm2.bias.data[mask]
        new_block.norm1.eps = block.norm1.eps
        new_block.norm2.eps = block.norm2.eps
        new_block.norm1.normalized_shape = (block.norm1.out_features_opt,)
        new_block.norm2.normalized_shape = (block.norm2.out_features_opt,)
        return new_block

class VIT(nn.Module):
    def __init__(self, image_size, patch_size, n_layers, n_heads, d_model, dropout, n_classes, bias):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.dropout = dropout
        self.n_classes = n_classes

        self.image_size = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        self.patch_embedding = PatchEmbedding(patch_size, d_model, bias = False)
        self.num_patches = (self.image_size[0] // self.patch_size[0]) * (self.image_size[1] // self.patch_size[1])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches+1, d_model))
        self.embed_dropout = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[Block(d_model) for _ in range(n_layers)])
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

    @staticmethod
    def from_pit(vit: pnn.PITVIT):
        hid_dim = vit.patch_embedding.out_features_opt
        hid_mask = vit.patch_embedding.features_mask.bool()
        has_bias = vit.head.bias is not None
        p = vit.head_dropout.p
        new_vit = VIT(vit.image_size, vit.patch_size, vit.n_layers, vit.n_heads, hid_dim, p, vit.n_classes, has_bias)
        new_vit.patch_embedding = PatchEmbedding.from_pit(vit.patch_embedding)
        new_vit.cls_token.data = vit.cls_token.data[:, :, hid_mask]
        new_vit.pos_embedding.data = vit.pos_embedding.data[:, :, hid_mask]
        new_vit.embed_dropout.p = p
        new_vit.head_dropout.p = p
        new_vit.blocks = nn.Sequential(*list(map(lambda b: Block.from_pit(b), vit.blocks)))
        new_vit.norm.weight.data = vit.norm.weight.data[hid_mask]
        new_vit.norm.bias.data = vit.norm.bias.data[hid_mask]
        new_vit.norm.bias.eps = vit.norm.eps
        new_vit.norm.normalized_shape = (vit.norm.out_features_opt,)
        new_vit.head.weight.data = mask_weight(vit.head.weight.data, hid_mask, torch.ones(vit.n_classes).bool())
        new_vit.head.bias.data = vit.head.bias.data
        return new_vit

def mask_weight(weight, in_mask, out_mask):
    weight = weight[out_mask]
    return weight[:, in_mask]
