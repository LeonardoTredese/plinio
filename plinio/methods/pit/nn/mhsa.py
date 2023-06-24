from typing import Dict, Any, Optional, cast, Iterator, Tuple, Final
import torch
import torch.nn as nn
import torch.fx as fx
from plinio.graph.features_calculation import ConstFeaturesCalculator, FeaturesCalculator, ModAttrFeaturesCalculator
from .features_masker import PITFeaturesMasker
from .binarizer import PITBinarizer
from .module import PITModule
from .linear import PITLinear

class PITMHSA(nn.Module, PITModule):
    def __init__(self,
                 hidden_dim: int,
                 n_heads: int,
                 qk_features_masker: PITFeaturesMasker,
                 v_features_masker: PITFeaturesMasker,
                 output_features_masker: PITFeaturesMasker,
                 binarization_threshold: float = 0.5,
                 ):
        super(PITMHSA, self).__init__()
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // n_heads
        assert self.head_dim * n_heads == hidden_dim, "hidden_dim must be divisible by n_heads"
        self.scale_factor = hidden_dim ** -.5
        self.binarization_threshold = binarization_threshold
        self.n_heads = n_heads
        self.qk_features_masker = qk_features_masker
        self.v_features_masker = v_features_masker
        self.output_features_masker = output_features_masker
        
        self.q_proj = PITLinear(nn.Linear(hidden_dim, hidden_dim, bias=False), \
                                qk_features_masker, binarization_threshold=binarization_threshold)
        self.k_proj = PITLinear(nn.Linear(hidden_dim, hidden_dim, bias=False), \
                                qk_features_masker, binarization_threshold=binarization_threshold)
        self.v_proj = PITLinear(nn.Linear(hidden_dim, hidden_dim, bias=False), \
                                v_features_masker, binarization_threshold=binarization_threshold)
        self.out_proj = PITLinear(nn.Linear(hidden_dim, hidden_dim, bias=True), \
                                  output_features_masker, binarization_threshold=binarization_threshold)
        self.input_features_calculator = ConstFeaturesCalculator(hidden_dim)
        v_proj_calculator = ModAttrFeaturesCalculator(self.v_proj, 'out_features_opt', 'features_mask')
        self.out_proj.input_features_calculator = v_proj_calculator

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape(*q.shape[:-1], self.n_heads, self.head_dim).transpose(-2, -3)
        k = k.reshape(*k.shape[:-1], self.n_heads, self.head_dim).transpose(-2, -3)
        v = v.reshape(*v.shape[:-1], self.n_heads, self.head_dim).transpose(-2, -3)

        attention = torch.matmul(q, k.transpose(-2, -1)) * self.scale_factor
        attention = torch.softmax(attention, dim=-1)

        x = torch.matmul(attention, v).reshape(*x.shape)
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
        input_features_calculator.register(self)
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

def attention_to_pit_attention(attention: Attention) -> PITMHSA:
    hidden_dim = attention.num_heads * attention.head_dim
    qk_features_masker = PITFeaturesMasker(hidden_dim)
    v_features_masker = PITFeaturesMasker(hidden_dim)
    output_features_masker = PITFeaturesMasker(hidden_dim)
    pit = PITMHSA(hidden_dim, attention.num_heads, qk_features_masker, v_features_masker, output_features_masker)
    pit.q_proj.weight.data, pit.k_proj.weight.data, pit.v_proj.weight.data= attention.qkv.weight.data.chunk(3)
    pit.out_proj.weight.data = attention.proj.weight.data
    pit.out_proj.bias.data = attention.proj.bias.data
    return pit

def test_attention_to_pit_attention():
    hidden_dim = 5
    seq_len = 3
    n_heads = 1
    batch_size = 1

    attention = Attention(hidden_dim, n_heads)
    pit_attention = attention_to_pit_attention(attention)

    attention.eval()
    pit_attention.eval()

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

    pit_attention = PITMHSA(hidden_dim, n_heads, qk_features_masker, v_features_masker, output_features_masker)
    timm_attention = Attention(hidden_dim, n_heads)

    pit_attention.eval()
    timm_attention.eval()

    # init all weights and biases to be 1
    for p in pit_attention.parameters():
        p.data.fill_(1)
    for p in timm_attention.parameters():
        p.data.fill_(1)

    with torch.no_grad():
        x = torch.randn(batch_size, seq_len, hidden_dim)
        pit_attention_output = pit_attention(x)
        timm_attention_output = timm_attention(x)
        assert torch.allclose(pit_attention_output, timm_attention_output), f"outputs should be the same"
        
def test_qk_same_mask():
    hidden_dim = 384
    n_heads = 12
    seq_len = 20
    batch_size = 32
    epochs = 100

    x = torch.randn(batch_size, seq_len, hidden_dim)
    y = torch.randn(batch_size, seq_len, hidden_dim)

    qk_features_masker = PITFeaturesMasker(hidden_dim)
    v_features_masker = PITFeaturesMasker(hidden_dim)
    output_features_masker = PITFeaturesMasker(hidden_dim)

    attention = PITMHSA(hidden_dim, n_heads, qk_features_masker, v_features_masker, output_features_masker)
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
    qk_features_masker = PITFeaturesMasker(hidden_dim)
    v_features_masker = PITFeaturesMasker(hidden_dim)
    output_features_masker = PITFeaturesMasker(hidden_dim)

    attention = PITMHSA(hidden_dim, n_heads, qk_features_masker, v_features_masker, output_features_masker)
    loss_fn = nn.CrossEntropyLoss()

    x = attention(x)
    x = classification_head(x[:, 0])
    loss = loss_fn(x, y)
    loss.backward()
    for name, param in attention.named_parameters():
        assert param.grad is not None, f"param {name} has no gradient"
        assert torch.any(param.grad != 0), f"param {name} has all zeros gradient"

def test_nas_gradient():
    hidden_dim = 384
    n_heads = 12
    seq_len = 20
    batch_size = 32
    epochs = 100

    classification_head = nn.Linear(hidden_dim, 4)

    qk_features_masker = PITFeaturesMasker(hidden_dim)
    v_features_masker = PITFeaturesMasker(hidden_dim)
    output_features_masker = PITFeaturesMasker(hidden_dim)

    attention = PITMHSA(hidden_dim, n_heads, qk_features_masker, v_features_masker, output_features_masker)
    
    x = torch.randn(batch_size, seq_len, hidden_dim)
    x = attention(x)

    loss = attention.get_size()
    loss.backward()
    nas_names = {name for name, _ in attention.named_nas_parameters()}
    for name, param in attention.named_nas_parameters():
        assert param.grad is not None, f"param {name} has no gradient"
        assert torch.any(param.grad != 0), f"param {name} has all zeros gradient"
    for name, param in attention.named_parameters():
        if name not in nas_names:
            assert param.grad is None or torch.all(param.grad == 0), f"param {name} aren't all zeros gradient" 

