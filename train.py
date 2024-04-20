import os
from dataclasses import dataclass, asdict
from functools import partial

import mlx.core as mx
import mlx.optimizers as optim
from mlx import nn
from mlx.utils import tree_flatten, tree_map
from tqdm import tqdm
import wandb

from dataset import config_dataloader
from llama import init_params, LLaMAConfig, LLaMA


@dataclass
class TrainerConfig:
    bsz: int = 8
    lr: float = 1e-4
    n_steps: int = 500
    grad_acc: int = 8
    warmup_steps: int = 50     # 10%
    pad_token_id: int = 65535  # Max value of uint16
    ckpt_name: str = 'mini-llama-wikitext-bsl'


def build_lr_schedule(cfg_t):
    warmup = optim.schedulers.linear_schedule(0.0, cfg_t.lr, cfg_t.warmup_steps)
    decay = optim.cosine_decay(cfg_t.lr, cfg_t.n_steps-cfg_t.warmup_steps)
    lr_schedule = optim.join_schedules([warmup, decay], [cfg_t.warmup_steps])
    return lr_schedule


def train():
    cfg_t = TrainerConfig()
    cfg_m = LLaMAConfig(n_layers=6, d_embd=512, n_heads=8)
    wandb.init(project='mini-llama-mlx', config={**asdict(cfg_t), **asdict(cfg_m)})

    dataloader = config_dataloader(seq_len=cfg_m.seq_len, **asdict(cfg_t))
    model = init_params(LLaMA(**asdict(cfg_m)))
    optimizer = optim.AdamW(learning_rate=build_lr_schedule(cfg_t))


    def train_forward_pass(model_, inputs_BT, labels_BT):
        pad_mask_BT = (inputs_BT != cfg_t.pad_token_id)
        logits_BTV = model_(inputs_BT * pad_mask_BT)
        logprobs_BT = nn.losses.cross_entropy(logits_BTV, labels_BT)
        loss = (logprobs_BT * pad_mask_BT).sum() / pad_mask_BT.sum()
        return loss


    state = [model.state, optimizer.state]
    @partial(mx.compile, inputs=state, outputs=state)
    def train_step(inputs_BT, labels_BT, loss, grads):
        loss_and_grad = nn.value_and_grad(model, train_forward_pass)
        mb_loss, mb_grads = loss_and_grad(model, inputs_BT, labels_BT)  # micro-batch
        loss = loss + mb_loss / cfg_t.grad_acc
        grads = tree_map(lambda g, mbg: (g + mbg / cfg_t.grad_acc), grads, mb_grads)
        return loss, grads


    model.train()
    update_steps = cfg_t.n_steps * cfg_t.grad_acc
    pbar = tqdm(total=update_steps)
    loss, grads = mx.zeros(1), tree_map(lambda p: mx.zeros(p.shape), model)

    for (inputs_BT, labels_BT), step_idx in zip(dataloader, range(update_steps)):
        try:
            loss, grads = train_step(inputs_BT, labels_BT, loss, grads)
            mx.eval(loss, grads)

            loss, lr = map(lambda x: x.item(), [loss, optimizer.learning_rate])
            pbar.set_description(f'{loss=:.4f} | {lr=:.2e}')
            pbar.update(1)

            if ((step_idx + 1) % cfg_t.grad_acc == 0) or (step_idx == update_steps - 1):
                optimizer.update(model, grads)
                mx.eval(state)
                wandb.log({'loss': loss, 'learning_rate': lr})
                loss, grads = mx.zeros(1), tree_map(lambda p: mx.zeros(p.shape), model)
        except KeyboardInterrupt:
            break

    pbar.close()
    mx.save_safetensors(cfg_t.ckpt_name, dict(tree_flatten(model)))


if __name__ == '__main__':
    wandb.login(key=os.environ.get('WANDB_KEY', None))  # Set WANDB_MODE="disabled" to disable
    train()
