from dataclasses import dataclass, asdict
from functools import partial
import time

import mlx.core as mx
import mlx.optimizers as optim
from mlx import nn

from llama import LLaMAConfig, LLaMA


@dataclass
class TrainerConfig:
    bsz: int = 16
    lr: float = 1e-4
    n_steps: int = 1000
    pad_token_id: int = 65535  # Max value of uint16


def train():
    # mx.set_default_device(mx.gpu)
    cfg_t = TrainerConfig()
    cfg_m = LLaMAConfig()

    model = LLaMA(**asdict(cfg_m))
    optimizer = optim.AdamW(learning_rate=cfg_t.lr)

    def forward(model_, inputs, targets):
        pad_mask = (inputs != cfg_t.pad_token_id)
        logits = model_(inputs * pad_mask)

        logprobs = nn.losses.cross_entropy(logits, targets)
        logprobs_m = logprobs * pad_mask
        loss = logprobs_m.sum() / pad_mask.sum()

        return loss

    state = [model.state, optimizer.state]
    @partial(mx.compile, inputs=state, outputs=state)
    def train_step(inputs, targets):
        loss_and_grad = nn.value_and_grad(model, forward)
        loss, grads = loss_and_grad(model, inputs, targets)
        optimizer.update(model, grads)
        return loss

    mx.random.seed(3985)
    input_idxs = mx.random.randint(0, cfg_m.vocab_size, shape=[cfg_t.bsz, cfg_m.seq_len])
    label_idxs = mx.random.randint(0, cfg_m.vocab_size, shape=[cfg_t.bsz, cfg_m.seq_len])
    for _ in range(2):
        tic = time.perf_counter()
        loss = train_step(input_idxs, label_idxs).item()
        print(f'{loss=:.4f}')
        toc = time.perf_counter()
        print(f'{(toc-tic):.4f} s')


if __name__ == '__main__':
    train()
