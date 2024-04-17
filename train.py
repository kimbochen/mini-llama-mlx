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
    n_steps: int = 100
    warmup_steps: int = 10
    pad_token_id: int = 65535  # Max value of uint16


def build_lr_schedule(cfg_t):
    warmup = optim.schedulers.linear_schedule(0.0, cfg_t.lr, cfg_t.warmup_steps)
    decay = optim.cosine_decay(cfg_t.lr, cfg_t.n_steps-cfg_t.warmup_steps)
    lr_schedule = optim.join_schedules([warmup, decay], [cfg_t.warmup_steps])
    return lr_schedule


def train():
    mx.random.seed(3985)
    cfg_t = TrainerConfig()
    cfg_m = LLaMAConfig(n_layers=6, d_embd=512, n_heads=8)

    dataloader = config_dataloader(seq_len=cfg_m.seq_len, **asdict(cfg_t))
    model = LLaMA(**asdict(cfg_m))
    model.init_params(cfg_m)
    optimizer = optim.AdamW(learning_rate=build_lr_schedule(cfg_t))


    def train_forward_pass(model_, inputs_BT, labels_BT):
        pad_mask_BT = (inputs_BT != cfg_t.pad_token_id)
        logits_BTV = model_(inputs_BT * pad_mask_BT)
        logprobs_BT = nn.losses.cross_entropy(logits_BTV, labels_BT)
        loss = (logprobs_BT * pad_mask_BT).sum() / pad_mask_BT.sum()
        return loss


    state = [model.state, optimizer.state]
    @partial(mx.compile, inputs=state, outputs=state)
    def train_step(inputs_BT, labels_BT):
        loss_and_grad = nn.value_and_grad(model, train_forward_pass)
        loss, grads = loss_and_grad(model, inputs_BT, labels_BT)
        optimizer.update(model, grads)
        ppl = loss.exp()
        return loss, ppl


    model.train()

    for inputs_BT, labels_BT in (pbar := tqdm(dataloader, total=cfg_t.n_steps)):
        loss, ppl = train_step(inputs_BT, labels_BT)
        mx.eval(state, loss, ppl)
        pbar.set_description((
            f'loss={loss.item():.4f} | ppl={ppl.item():.4f} | lr={optimizer.learning_rate.item():.2e} | '
            f'peak_mem={(mx.metal.get_peak_memory() / 2**30):.2f}G'
        ))


    mx.save_safetensors('mini-llama-wikitext', dict(tree_flatten(model)))


if __name__ == '__main__':
    train()
