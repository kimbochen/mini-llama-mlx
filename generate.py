import sys
from dataclasses import dataclass, asdict
from functools import partial

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_unflatten
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm

from dataset import config_dataloader
from llama import LLaMAConfig, LLaMA


@dataclass
class GenerationConfig:
    max_new_tokens: int = 256


def generate(prompt):
    tokenizer = SentencePieceProcessor(model_file='tokenizer.model')

    cfg_m = LLaMAConfig(n_layers=6, d_embd=512, n_heads=8)
    model = LLaMA(**asdict(cfg_m))
    model.update(tree_unflatten([*mx.load(sys.argv[1]).items()]))

    cfg_g = GenerationConfig()
    tokens_BT = mx.array([tokenizer.encode(prompt, add_bos=True)], dtype=mx.uint16)
    new_tokens = 0
    new_token_BT = None

    while new_tokens < cfg_g.max_new_tokens and new_token_BT != tokenizer.eos_id:
        logits_BTC = model(tokens_BT)
        new_token_BT = mx.argmax(logits_BTC[:, -1, :], axis=-1, keepdims=True)  # Greedy sampling
        tokens_BT = mx.concatenate([tokens_BT, new_token_BT], axis=-1)[:, :cfg_m.seq_len]
        new_tokens += 1

    completion = tokenizer.decode(tokens_BT.tolist())[0]
    print(completion)


if __name__ == '__main__':
    generate(sys.argv[2])
