from torch import nn

import torch
import math
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, block_size, vocab_size, n_layer, n_head, n_embd, bias=False, dropout=0.1):
        super().__init__()

        self.bs = block_size
        self.vs = vocab_size
        self.nl = n_layer
        self.nh = n_head
        self.ne = n_embd
        self.b = bias
        self.d = dropout

        assert self.ne % self.nh == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(self.ne, 3 * self.ne, bias=self.b)
        # output projection
        self.c_proj = nn.Linear(self.ne, self.ne, bias=self.b)
        # regularization
        self.attn_dropout = nn.Dropout(self.d)
        self.resid_dropout = nn.Dropout(self.d)
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(self.bs, self.bs)).view(1, 1, self.bs, self.bs))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.ne, dim=2)
        k = k.view(B, T, self.nh, C // self.nh).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.nh, C // self.nh).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.nh, C // self.nh).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.d if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, n_embd, bias, dropout):
        super().__init__()

        self.ne = n_embd
        self.b = bias
        self.d = dropout

        self.c_fc    = nn.Linear(self.ne, 4 * self.ne, bias=self.b)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * self.ne, self.ne, bias=self.b)
        self.dropout = nn.Dropout(self.d)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, block_size, vocab_size, n_layer, n_head, n_embd, bias=False, dropout=0.1):
        super().__init__()

        self.ne = n_embd
        self.b = bias

        self.ln_1 = LayerNorm(self.ne, self.b)
        self.attn = CausalSelfAttention(block_size, vocab_size, n_layer, n_head, n_embd, bias, dropout)
        self.ln_2 = LayerNorm(self.ne, self.b)
        self.mlp = MLP(self.ne, self.b, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT2Enc(nn.Module):
    def __init__(self, block_size, vocab_size, n_layer, n_head, n_embd, bias=False, dropout=0.1) -> None:
        super(GPT2Enc, self).__init__()

        self.bs = block_size
        self.vs = vocab_size
        self.nl = n_layer
        self.nh = n_head
        self.ne = n_embd
        self.b = bias
        self.d = dropout
    
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(self.vs, self.ne),
            wpe = nn.Embedding(self.bs, self.ne),
            drop = nn.Dropout(self.d),
            h = nn.ModuleList([Block(self.bs, self.vs, self.nl, self.nh, self.ne, self.b, self.d) for _ in range(self.nl)]),
            ln_f = LayerNorm(self.ne, bias=self.b),
        ))

        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * self.nl))
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx):
        device = idx.device
        b, t = idx.size()
        assert t <= self.bs, f"Cannot forward sequence of length {t}, block size is only {self.bs}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        return x.mean(1)