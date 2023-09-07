import torch 
import torch.nn as nn
import timm
from typing import Final
from functools import partial
from plinio.methods.pit.nn import PITAttention, PITModule, PITLinear, PITMlp, PITBlock, PITPatchEmbedding, PITVIT
from plinio.graph.features_calculation import ConstFeaturesCalculator, FeaturesCalculator, ModAttrFeaturesCalculator
from plinio.methods.pit.nn.features_masker import PITFeaturesMasker
from plinio.methods.pit.nn.binarizer import PITBinarizer

# implementation of attention mechanism shown in timm
class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = False

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features: int =None,
            out_features: int =None,
            act_layer: nn.Module=nn.GELU,
            norm_layer: nn.Module=None,
            bias: bool=True,
            drop: float=0.,
            use_conv: bool=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features 
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias)
        self.act = act_layer()


        self.drop1 = nn.Dropout(drop)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class Block(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            mlp_layer=Mlp,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

### Tests ###

def test_copy_timm_attention():
    model = timm.create_model('vit_tiny_patch16_384', pretrained=True)
    for i, block in enumerate(model.blocks.children()):
        attention = block.attn
        hidden_dim = attention.num_heads * attention.head_dim
        pit_attention = PITAttention.from_timm(attention)
        attention.eval()
        pit_attention.eval()
        x = torch.randn(1, (384 // 16) ** 2  + 1, hidden_dim)
        assert torch.allclose(attention(x), pit_attention(x)), f"Attention {i} faialed"

def test_copy_timm_mlp():
    model = timm.create_model('vit_tiny_patch16_384', pretrained=True)
    for i, block in enumerate(model.blocks.children()):
        mlp = block.mlp
        pit_mlp = PITMlp.from_timm(mlp)
        in_dim = mlp.fc1.weight.shape[1]
        x = torch.randn(1, (384 // 16) ** 2  + 1, in_dim)
        mlp.eval()
        pit_mlp.eval()
        assert torch.allclose(mlp(x), pit_mlp(x)), f"MLP {i} failed"

def test_copy_timm_block():
    model = timm.create_model('vit_tiny_patch16_384', pretrained=True)
    for i, block in enumerate(model.blocks.children()):
        pit_block = PITBlock.from_timm(block)
        in_dim = block.mlp.fc1.weight.shape[1]
        x = torch.randn(1, (384 // 16) ** 2  + 1, in_dim)
        block.eval()
        pit_block.eval()
        assert torch.allclose(block(x), pit_block(x)), f"Block {i} failed"

def test_copy_timm_embed():
    model = timm.create_model('vit_tiny_patch16_384', pretrained=True)
    pit_embed = PITPatchEmbedding.from_timm(model.patch_embed, (384,) * 2)
    x = torch.randn(1, 3, 384, 384)
    pit_embed.eval()
    model.patch_embed.eval()
    assert torch.allclose(pit_embed(x), model.patch_embed(x)), "Patch embedding failed"


def test_vit_to_pit():
    model = timm.create_model('vit_tiny_patch16_384', pretrained=True)
    pit_model = PITVIT.from_timm(model, (384,)*2)
    x = torch.randn(1, 3, 384, 384)
    x_tokens = torch.ones(1, (384 // 16) ** 2 + 1, 192)
    x_pre_tokens = torch.ones(1, (384 // 16) ** 2, 192)
    pit_model.eval()
    model.eval()
    assert torch.allclose(model.patch_embed(x), pit_model.patch_embedding(x)), "VIT and PITVIT should have the same patch embedding"
    assert torch.allclose(model.cls_token, pit_model.cls_token), "VIT and PITVIT should have the same cls token"
    assert torch.allclose(model.pos_embed, pit_model.pos_embedding), "VIT and PITVIT should have the same position embedding"
    assert torch.allclose(model._pos_embed(x_pre_tokens), pit_model._pos_embed(x_pre_tokens)), "VIT and PITVIT should have the same position embedding & cls token"
    for i, block in enumerate(model.blocks.children()):
        assert torch.allclose(block(x_tokens), pit_model.blocks[i](x_tokens)), f"VIT and PITVIT should have the same block {i}"
    assert torch.allclose(model.blocks(x_tokens), pit_model.blocks(x_tokens)), "VIT and PITVIT should have the same blocks"
    assert torch.allclose(model.norm(x_tokens), pit_model.norm(x_tokens)), "VIT and PITVIT should have the same norm"
    assert torch.allclose(model.head(x_tokens[:, 0]), pit_model.head(x_tokens[:, 0])), "VIT and PITVIT should have the same head"
    assert torch.allclose(model.forward_features(x), pit_model.forward_features(x)), "VIT and PITVIT should have the same forward features"
    assert torch.allclose(model(x), pit_model(x)), "VIT and PITVIT should have the same output"

def test_attention_to_pit():
    hidden_dim = 5
    seq_len = 3
    n_heads = 1
    batch_size = 1

    attention = Attention(hidden_dim, n_heads)
    pit_attention = PITAttention.from_timm(attention)

    attention.eval()
    pit_attention.eval()
    
    assert attention.scale == pit_attention.scale_factor, "Different scales factor"
    with torch.no_grad():
        x = torch.randn(batch_size, seq_len, hidden_dim)
        attention_output = attention(x)
        pit_attention_output = pit_attention(x)
        assert torch.allclose(attention_output, pit_attention_output), 'attention output and pit attention output should be equal'


def test_attention_output():
    hidden_dim = 5
    seq_len = 3
    n_heads = 1
    batch_size = 1
    
    qk_features_masker = PITFeaturesMasker(hidden_dim)
    v_features_masker = PITFeaturesMasker(hidden_dim)
    output_features_masker = PITFeaturesMasker(hidden_dim)

    pit_attention = PITAttention(hidden_dim, n_heads, out_bias = True)
    timm_attention = Attention(hidden_dim, n_heads)

    pit_attention.eval()
    timm_attention.eval()

    # init all weights and biases to be 1
    for p in pit_attention.parameters():
        p.data.fill_(1)
    for p in timm_attention.parameters():
        p.data.fill_(1)

    with torch.no_grad():
        x = torch.ones(batch_size, seq_len, hidden_dim)
        pit_attention_output = pit_attention(x)
        timm_attention_output = timm_attention(x)
        assert torch.allclose(pit_attention_output, timm_attention_output), f"outputs should be the same, Timm: \n {timm_attention_output} \n PIT: \n {pit_attention_output}"
        
def test_hiddendim_embed_features_mask():
    image_size = 384
    patch_size = 16
    n_layers = 12
    n_heads = 12
    d_model = 384
    ff_scale = 4
    dropout = 0.1
    n_classes = 4
    bias = True

    x = torch.randn(1, 3, image_size, image_size)

    model = PITVIT(image_size, patch_size, n_layers, n_heads, d_model, ff_scale, dropout, n_classes, bias)
    model.hidden_dim_wiring()
    random_mask = nn.Parameter(torch.randint(0, 2, (d_model,), dtype=torch.float32))
    model.patch_embedding.conv.out_features_masker.alpha = random_mask
    mask = model.patch_embedding.conv.features_mask == 0
    x = model.patch_embedding(x)
    assert torch.all(x[..., mask] == 0), f"masked patch embedding should be zeros insetad of {x[..., mask]}"
    x = model._pos_embed(x)
    assert torch.all(x[..., mask] == 0), f"masked pos_embed should be zeros insetad of {model._pos_embed[..., mask]}"

def test_weight_gradient():
    hidden_dim = 384
    n_heads = 12
    seq_len = 20
    batch_size = 32
    x = torch.randn(batch_size, seq_len, hidden_dim)
    y = torch.tensor([0]*batch_size)
    classification_head = nn.Linear(hidden_dim, 4)
    attention = PITAttention(hidden_dim, n_heads) 
    loss_fn = nn.CrossEntropyLoss()
    x = attention(x)
    x = classification_head(x[:, 0])
    loss = loss_fn(x, y)
    loss.backward()
    for name, param in attention.named_parameters():
        assert param.grad is not None, f"param {name} has no gradient"
        assert torch.any(param.grad != 0), f"param {name} has all zeros gradient"

def test_nas_gradient():
    x = torch.randn(1, 3, 384, 384)
    model = timm.create_model('vit_tiny_patch16_384', pretrained=True, num_classes=10)
    pit_model = PITVIT.from_timm(model, x.shape[-2:])
    pit_model.cascade_wiring()
    pit_model(x)
    loss = pit_model.get_size()
    loss.backward()
    nas_names = {name for name, _ in pit_model.named_nas_parameters()}
    for name, param in pit_model.named_nas_parameters():
        assert param.grad is not None, f"param {name} has no gradient"
        assert torch.any(param.grad != 0), f"param {name} has all zeros gradient"
    for name, param in pit_model.named_parameters():
        if name not in nas_names:
            assert param.grad is None or torch.all(param.grad == 0), f"param {name} aren't all zeros gradient" 

def main():
    test_hiddendim_embed_features_mask()
    test_nas_gradient()
    test_weight_gradient()
    test_attention_output()
    test_attention_to_pit()
    test_copy_timm_attention()
    test_copy_timm_mlp()
    test_copy_timm_embed()
    test_copy_timm_block()
    test_vit_to_pit() 

if __name__ == "__main__":
    main()
