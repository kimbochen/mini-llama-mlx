from dataclasses import dataclass, asdict
from functools import partial

import mlx.core as mx
import mlx.optimizers as optim
from mlx import nn
from mlx.utils import tree_flatten
from tqdm import tqdm

from dataset import config_dataloader
from llama import LLaMAConfig, LLaMA


@dataclass
class TrainerConfig:
    bsz: int = 16
    lr: float = 1e-4
    n_steps: int = 800
    pad_token_id: int = 65535  # Max value of uint16


def forward(model, inputs, targets, pad_token_id):
    pad_mask = (inputs != pad_token_id)
    logits = model(inputs * pad_mask)

    logprobs = nn.losses.cross_entropy(logits, targets)
    logprobs_m = logprobs * pad_mask
    loss = logprobs_m.sum() / pad_mask.sum()

    return loss


def train():
    cfg_t = TrainerConfig()
    cfg_m = LLaMAConfig(n_layers=6, d_embd=512, n_heads=8)

    load_data = config_dataloader(cfg_t.bsz, cfg_m.seq_len, cfg_t.pad_token_id)
    model = LLaMA(**asdict(cfg_m))
    optimizer = optim.AdamW(learning_rate=cfg_t.lr)

    forward_fn = partial(forward, pad_token_id=cfg_t.pad_token_id)
    state = [model.state, optimizer.state]
    @partial(mx.compile, inputs=state, outputs=state)
    def train_step(inputs, targets):
        loss_and_grad = nn.value_and_grad(model, forward_fn)
        loss, grads = loss_and_grad(model, inputs, targets)
        optimizer.update(model, grads)
        return loss

    for inputs, labels in (pbar := tqdm(load_data(cfg_t.n_steps), total=cfg_t.n_steps)):
        loss = train_step(inputs, labels)
        mx.eval(state, loss)
        pbar.set_description(f'loss={loss.item():.4f} | peak_mem={(mx.metal.get_peak_memory() / 2**30):.2f}GB')

    mx.save_safetensors('mini-llama-wikitext', dict(tree_flatten(model)))


if __name__ == '__main__':
    train()
