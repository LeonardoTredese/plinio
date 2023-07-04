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

def test_copy_timm_attention():
    model = timm.create_model('vit_tiny_patch16_384', pretrained=True)
    for i, block in enumerate(model.blocks.children()):
        attention = block.attn
        hidden_dim = attention.num_heads * attention.head_dim
        pit_attention = attention_to_pit(attention)
        attention.eval()
        pit_attention.eval()
        x = torch.randn(1, (384 // 16) ** 2  + 1, hidden_dim)
        assert torch.allclose(attention(x), pit_attention(x)), f"Attention {i} faialed"

def test_copy_timm_mlp():
    model = timm.create_model('vit_tiny_patch16_384', pretrained=True)
    for i, block in enumerate(model.blocks.children()):
        mlp = block.mlp
        pit_mlp = mlp_to_pit(mlp)
        in_dim = mlp.fc1.weight.shape[1]
        x = torch.randn(1, (384 // 16) ** 2  + 1, in_dim)
        mlp.eval()
        pit_mlp.eval()
        assert torch.allclose(mlp(x), pit_mlp(x)), f"MLP {i} failed"

def test_copy_timm_block():
    model = timm.create_model('vit_tiny_patch16_384', pretrained=True)
    for i, block in enumerate(model.blocks.children()):
        pit_block = block_to_pit(block)
        in_dim = block.mlp.fc1.weight.shape[1]
        x = torch.randn(1, (384 // 16) ** 2  + 1, in_dim)
        block.eval()
        pit_block.eval()
        assert torch.allclose(block(x), pit_block(x)), f"Block {i} failed"

def test_copy_timm_embed():
    model = timm.create_model('vit_tiny_patch16_384', pretrained=True)
    pit_embed = embed_to_pit(model.patch_embed, (384,) * 2)
    x = torch.randn(1, 3, 384, 384)
    pit_embed.eval()
    model.patch_embed.eval()
    assert torch.allclose(pit_embed(x), model.patch_embed(x)), "Patch embedding failed"

def block_to_pit(block: Block) -> PITBlock:
    dim = block.attn.num_heads * block.attn.head_dim
    pit_block = PITBlock(dim, block.attn.num_heads)
    pit_block.attn = attention_to_pit(block.attn)
    pit_block.mlp = mlp_to_pit(block.mlp)
    pit_block.norm1.load_state_dict(block.norm1.state_dict())
    pit_block.norm2.load_state_dict(block.norm2.state_dict())
    pit_block.norm1.eps = block.norm1.eps
    pit_block.norm2.eps = block.norm2.eps
    return pit_block

def mlp_to_pit(mlp: Mlp) -> PITModule:
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

def attention_to_pit(attention: Attention) -> PITAttention:
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

def embed_to_pit(patch_embed, image_dimension):
    patch_size = patch_embed.proj.kernel_size
    hidden_dim = patch_embed.proj.weight.shape[0]
    has_bias = patch_embed.proj.bias is not None
    pit_embed = PITPatchEmbedding(image_dimension, patch_size, hidden_dim, bias = has_bias)
    pit_embed.conv.weight.data = patch_embed.proj.weight.data
    if has_bias:
        pit_embed.conv.bias.data = patch_embed.proj.bias.data
    return pit_embed

def vit_to_pit(vit, image_size):
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
    pit_vit.patch_embedding = embed_to_pit(vit.patch_embed, image_size)
    embed_features_calculator = ModAttrFeaturesCalculator(pit_vit.patch_embedding, 'out_features_opt', 'features_mask')
    for i, block in enumerate(vit.blocks.children()):
        pit_vit.blocks[i] = block_to_pit(block)
        pit_vit.blocks[i].input_features_calculator = embed_features_calculator
    pit_vit.norm.load_state_dict(vit.norm.state_dict())
    pit_vit.norm.eps = vit.norm.eps
    pit_vit.head.weight.data = vit.head.weight.data
    if vit.head.bias is not None:
        pit_vit.head.bias.data = vit.head.bias.data
    return pit_vit

### Tests ###

def test_vit_to_pit():
    model = timm.create_model('vit_tiny_patch16_384', pretrained=True)
    pit_model = vit_to_pit(model, (384,) * 2)
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
    pit_attention = attention_to_pit(attention)

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
        
def test_qk_same_mask():
    hidden_dim = 384
    n_heads = 12
    seq_len = 20
    batch_size = 32
    epochs = 100

    x = torch.randn(batch_size, seq_len, hidden_dim)
    y = torch.randn(batch_size, seq_len, hidden_dim)

    attention = PITAttention(hidden_dim, n_heads)
    # impose random masks
    attention.qk_features_masker.alpha = nn.Parameter(torch.randint(0, 2, (hidden_dim,), dtype=torch.float32))
    with torch.no_grad():
        q_hat = attention.q_proj(x)
        k_hat = attention.k_proj(x)
        mask = attention.qk_features_mask == 0
        # assert that all the masked features are zero
        assert torch.all(q_hat[..., mask] == 0), f"masked q_hat should be zeros insetad of {q_hat[..., mask]}"
        assert torch.all(k_hat[..., mask] == 0), f"masked k_hat should be zeros insetad of {k_hat[..., mask]}"

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
    pit_model = vit_to_pit(model, x.shape[-2:])
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
    test_nas_gradient()
    test_qk_same_mask()
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
