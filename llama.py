from dataclasses import dataclass, asdict

import mlx.core as mx
from mlx import nn
from mlx.nn import MultiHeadAttention as MHA
from mlx.core.fast import scaled_dot_product_attention



@dataclass
class LLaMAConfig:
    '''
    Model configuration reference:
    https://github.com/epfml/llm-baselines?tab=readme-ov-file#results-on-wikitext
    '''
    n_layers: int = 24
    vocab_size: int = 32000  # V , LLaMA 2 tokenizer
    d_embd: int = 768        # D
    n_heads: int = 12        # N
    seq_len: int = 512       # T
    rope_theta: float = 1e4
    rope_scale: float = 1.0
    ffn_mult: int = 256
    norm_eps: float = 1e-5


class SelfAttention(nn.Module):
    def __init__(self, d_embd, n_heads, rope_theta, rope_scale, **kwargs):
        super().__init__()
        assert d_embd % n_heads == 0
        self.d_head = d_embd // n_heads  # H

        self.attn_proj = nn.Linear(d_embd, 3*d_embd, bias=False)
        self.rope = nn.RoPE(self.d_head, base=rope_theta, scale=rope_scale)
        self.scale = self.d_head ** -0.5
        self.out_proj = nn.Linear(d_embd, d_embd, bias=False)

    def __call__(self, x_BTD, mask_TT):
        B, T, D = x_BTD.shape
        to_attn_heads = lambda z: z.reshape(B, T, -1, self.d_head).transpose(0, 2, 1, 3)

        qkv_BTD = self.attn_proj(x_BTD).split(3, axis=-1)
        Q_BNTH, K_BNTH, V_BNTH = map(to_attn_heads, qkv_BTD)
        Q_BNTH, K_BNTH = self.rope(Q_BNTH), self.rope(K_BNTH)
        O_BNTH = scaled_dot_product_attention(Q_BNTH, K_BNTH, V_BNTH, scale=self.scale, mask=mask_TT)
        out_BTD = self.out_proj(O_BNTH.transpose(0, 2, 1, 3).reshape(B, T, D))

        return out_BTD

    def init_params(self, cfg):
        self.attn_proj.apply(nn.init.normal(0.0, cfg.d_embd**0.5))
        self.out_proj.apply(nn.init.normal(0.0, cfg.d_embd**0.5))


class FeedForwardNet(nn.Module):
    def __init__(self, d_embd, ffn_mult, **kwargs):
        super().__init__()
        hidden_dim = int((4 * d_embd) * 2 / 3)  # C
        self.hidden_dim = ffn_mult * ((hidden_dim + ffn_mult - 1) // ffn_mult)  # The next multiple of ffn_mult

        self.gate_proj = nn.Linear(d_embd, self.hidden_dim, bias=False)
        self.up_proj = nn.Linear(d_embd, self.hidden_dim, bias=False)
        self.down_proj = nn.Linear(self.hidden_dim, d_embd, bias=False)

    def __call__(self, x_BTD):
        h_BTC = nn.silu(self.gate_proj(x_BTD)) * self.up_proj(x_BTD)  # SwiGLU
        out_BTD = self.down_proj(h_BTC)
        return out_BTD

    def init_params(self, cfg):
        self.gate_proj.apply(nn.init.normal(0.0, cfg.d_embd**-0.5))
        self.up_proj.apply(nn.init.normal(0.0, cfg.d_embd**-0.5))
        self.down_proj.apply(nn.init.normal(0.0, self.hidden_dim**-0.5))


class TransformerBlock(nn.Module):
    def __init__(self, d_embd, norm_eps, **kwargs):
        super().__init__()
        self.pre_norm = nn.RMSNorm(d_embd, norm_eps)
        self.attn = SelfAttention(d_embd=d_embd, **kwargs)
        self.ffn_norm = nn.RMSNorm(d_embd, norm_eps)
        self.ffn = FeedForwardNet(d_embd=d_embd, **kwargs)

    def __call__(self, x_BTD, mask_TT):
        h_BTD = x_BTD + self.attn(self.pre_norm(x_BTD), mask_TT)
        out_BTD = h_BTD + self.ffn(self.ffn_norm(h_BTD))
        return out_BTD

    def init_params(self, cfg):
        self.pre_norm.apply(nn.init.constant(1.0))
        self.attn.init_params(cfg)
        self.ffn_norm.apply(nn.init.constant(1.0))
        self.ffn.init_params(cfg)


class LLaMA(nn.Module):
    def __init__(self, vocab_size, n_layers, d_embd, norm_eps, **kwargs):
        super().__init__()
        self.embd_toks = nn.Embedding(vocab_size, d_embd)
        self.layers = [
            TransformerBlock(d_embd=d_embd, norm_eps=norm_eps,**kwargs)
            for _ in range(n_layers)
        ]
        self.out_norm = nn.RMSNorm(d_embd, norm_eps)
        self.lm_head = nn.Linear(d_embd, vocab_size, bias=False)

    def __call__(self, tok_idxs_BT):
        h_BTD = self.embd_toks(tok_idxs_BT)

        causal_mask_TT = MHA.create_additive_causal_mask(h_BTD.shape[1])
        for layer in self.layers:
            h_BTD = layer(h_BTD, causal_mask_TT)
        h_BTD = self.out_norm(h_BTD)

        logits_BTV = self.lm_head(h_BTD)

        return logits_BTV

    def init_params(self, cfg):
        self.embd_toks.apply(nn.init.normal(0.0, 1.0))
        for layer in self.layers:
            layer.init_params(cfg)
        self.out_norm.apply(nn.init.constant(1.0))
        self.lm_head.apply(nn.init.normal(0.0, cfg.d_embd**-0.5))


if __name__ == '__main__':
    cfg_m = LLaMAConfig()
    model = LLaMA(**asdict(cfg_m))

    mx.random.seed(3985)
    tok_idxs = mx.random.randint(0, cfg_m.vocab_size, shape=[2, cfg_m.seq_len])

    print(model(tok_idxs).shape)
