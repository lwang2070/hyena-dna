""" Wrapper around nn.MultiheadAttention to adhere to SequenceModule interface. """

import torch
import torch.nn.functional as F
from torch import nn
import hydra
from src.models.sequence.base import SequenceModule, TransposedModule
import src.models.nn.utils as U
from einops import rearrange

@TransposedModule
class MultiheadAttention(SequenceModule):
    """ Simple wrapper for MultiheadAttention """
    def __init__(self, d_model, n_heads, *args, causal=True, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.d_output = d_model
        self.mha = nn.MultiheadAttention(d_model, n_heads, *args, batch_first=True, **kwargs)
        self.causal = causal

    def forward(self, src, attn_mask=None, key_padding_mask=None, state=None, **kwargs):
        """ state should represent a mask and key padding mask """
        if self.causal and attn_mask is None:
            attn_mask = torch.triu(torch.ones(src.size(-2), src.size(-2),
                                              dtype=torch.bool, device=src.device),
                                       diagonal=1)
        # attn_mask, key_padding_mask = state
        # Note that this returns None for the second argument
        y, _ = self.mha(src, src, src, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)
        return y, None

    def step(self, x, state):
        # TODO proper cached inference
        # x: (B, D)
        pass


class VitAttention(SequenceModule):
    """Copied from implementation for ViT: only used for ViT model

    This attention class makes several simplifying assumptions (commonly satisfied in vision
       applications):
    1. q = k = v
    2. No masks: no attention mask, no key padding mask
    3. Embed dimension = Input dimension, i.e. projection matrices are square.
    """

    @property
    def d_output(self):
        return self.dim

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.,
        # proj_drop=0.,
        packed_linear=True,
        linear_cfg=None,
        **kwargs,
    ):
        """packed_linear: whether to pack all 3 q_proj, k_proj, v_proj into 2 matrix.
        This option is to be compatible with T2T-ViT pretrained weights, where there's only one
        projection weight matrix.
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        if linear_cfg is not None:
            packed_linear = False
        self.packed_linear = packed_linear
        if packed_linear:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        else:
            if linear_cfg is None:
                linear_cfg = {'_target_': 'torch.nn.Linear'}
            self.q_proj = hydra.utils.instantiate(linear_cfg, dim, dim, bias=qkv_bias,
                                                  _recursive_=False)
            self.k_proj = hydra.utils.instantiate(linear_cfg, dim, dim, bias=qkv_bias,
                                                  _recursive_=False)
            self.v_proj = hydra.utils.instantiate(linear_cfg, dim, dim, bias=qkv_bias,
                                                  _recursive_=False)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        # Removing this dropout because we do this in SequenceResidualBlock
        # self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, state=None):
        B, N, C = x.shape
        if self.packed_linear:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        else:
            q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
            q, k, v = [rearrange(x, 'b n (h d) -> b h n d', h=self.num_heads) for x in (q, k, v)]

        # attn = (q @ k.transpose(-2, -1) * self.scale)
        # Use `torch.baddbmm` (a bit more efficient w/ alpha param for scaling -- from Megatron-LM)
        bsz, num_heads, q_seq_len, dk = q.size()
        _, _, k_seq_len, _ = k.size()
        q = rearrange(q, 'b h t d -> (b h) t d')
        k = rearrange(k, 'b h s d -> (b h) d s')
        # Preallocate attn_weights for `baddbmm`
        attn = torch.empty(bsz * num_heads, q_seq_len, k_seq_len, dtype=q.dtype, device=q.device)
        attn = rearrange(torch.baddbmm(attn, q, k, beta=0, alpha=self.scale),
                         '(b h) t s -> b h t s', h = self.num_heads)

        attn = F.softmax(attn, dim=-1, dtype=v.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        # x = self.proj_drop(x)
        return x, None

class MHABase(nn.Module):
    """A base class for MHA."""

    def __init__(
        self,
        d_model,
        num_heads,
        qk_dim=None,
        v_dim=None,
        bias=True,
        dropout=0,
        softmax_scale=None,
        *args,
        **kwargs,
    ):
        super().__init__()
        
        # Attributes
        if dropout > 0:
            raise NotImplementedError('Dropout is not yet implemented!')
        self.dropout = dropout
        self.num_heads = num_heads
        self.d_model = d_model
        self.softmax_scale = torch.sqrt(torch.tensor(d_model)) if softmax_scale is None else softmax_scale

        # Parameters
        self.qk_dim = d_model if qk_dim is None else qk_dim
        self.v_dim = d_model if v_dim is None else v_dim
        assert (
            self.qk_dim % num_heads == 0
        ), "qk dimension must be multiples of numbers of heads!"
        assert (
            self.v_dim % num_heads == 0
        ), "v dimension must be multiples of numbers of heads!"
        self.qkv_proj = nn.Linear(d_model, self.qk_dim * 2 + self.v_dim, bias=bias)
        self.o_proj = nn.Linear(self.v_dim, d_model, bias=bias)
    
    @property
    def d_output(self):
        return self.d_model

class CMHA(MHABase):
    
    def __init__(self, d_model, num_heads, chunk_size, pool_method, qk_dim=None, v_dim=None, bias=True, dropout=0, *args, **kwargs):
        # Init Base
        super().__init__(**{k:v for k, v in locals().items() if k != 'self'})
        
        # Attributes
        self.chunk_size = chunk_size
        assert pool_method in ['max_pool', 'mean_pool'], 'Pooling method must be either max_pool or mean_pool'
        self.pool_method = pool_method
    
    def forward(self, x, state=None):
        '''
        Input:
            x: (B, L, D)
        '''
        B, L, D = x.size()
        assert L % self.chunk_size == 0, 'Sequence length must be multiples of chunk_size.'
        N = L // self.chunk_size  # number of chunks in a sequence
        q, k, v = [rearrange(e, 'b l (h d) -> b l h d', h=self.num_heads) for e in torch.split(self.qkv_proj(x), [self.qk_dim, self.qk_dim, self.v_dim], dim=-1)]
        
        # Split into chunks (B, N, H, d) H is num_heads
        q_chunks = rearrange(q, 'b (n c) h d -> b n c h d', c=self.chunk_size)
        k_chunks = rearrange(k, 'b (n c) h d -> b n c h d', c=self.chunk_size)
        
        # Hierarchical attention        
        if self.pool_method == 'max_pool':
            q_pool, k_pool = q_chunks.max(dim=2)[0], k_chunks.max(dim=2)[0]  # (B, N, H, d)
        elif self.pool_method == 'mean_pool':
            q_pool, k_pool = q_chunks.mean(dim=2), k_chunks.mean(dim=2)
        else:
            raise NotImplemented(f'{self.pool_method} is not implemented!')
        
        token2chunk = torch.einsum('blhd,bnhd->bhln', q, k_pool)
        chunk2token = torch.einsum('bnhd,bnchd->bhnc', q_pool, k_chunks)
        A = rearrange(torch.einsum('bhln,bhnc->bhlnc', token2chunk, chunk2token), 'b h l n c -> b h l (n c)')  # (B, H, L, L)
        
        # Local direct attention
        direct_attn = torch.einsum('bnqhd,bnkhd->bhnqk', q_chunks, k_chunks)
        for i in range(N):
            start, end = i * self.chunk_size, (i+1) * self.chunk_size
            A[..., start:end, start:end] = direct_attn[:,:, i]
        
        # Causal mask
        mask = torch.triu(torch.ones(A.size(-1), A.size(-1),
                                              dtype=torch.bool, device=A.device),
                                       diagonal=1) == 0
        A = A * mask / self.softmax_scale
        A = torch.nn.functional.softmax(A, dim=-1)   
        return self.o_proj(rearrange(torch.einsum('bhqv,bvhd->bhqd', A, v), 'b h q d->b q (h d)'))