import os
from dataclasses import asdict, dataclass, field
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
    bsz: int = 16
    lr: float = 1e-3
    n_update_steps: int = 900
    grad_acc_steps: int = 16
    warmup_ratio: float = 0.1
    n_steps: int = field(init=False)
    warmup_steps: int = field(init=False)
    pad_token_id: int = -1
    ckpt_name: str = 'mini-llama-wikitext-bsl'

    def __post_init__(self):
        self.n_steps = self.n_update_steps * self.grad_acc_steps
        self.warmup_steps = int(self.n_update_steps * self.warmup_ratio)


def train():
    cfg_t = TrainerConfig()
    cfg_m = LLaMAConfig(n_layers=6, d_embd=256, n_heads=8)
    wandb.init(project='mini-llama-mlx', config={**asdict(cfg_t), **asdict(cfg_m)})

    load_train_data = config_dataloader(**asdict(cfg_t), **asdict(cfg_m), split='train')

    model = init_params(LLaMA(**asdict(cfg_m)))

    lr_schedule = optim.join_schedules([
        optim.schedulers.linear_schedule(0.0, cfg_t.lr, cfg_t.warmup_steps),
        optim.cosine_decay(cfg_t.lr, cfg_t.n_update_steps-cfg_t.warmup_steps)
    ], [cfg_t.warmup_steps])
    optimizer = optim.AdamW(learning_rate=lr_schedule)


    def train_forward_pass(model_, inputs_BT, labels_BT):
        pad_mask_BT = (inputs_BT != cfg_t.pad_token_id)
        logits_BTV = model_(inputs_BT * pad_mask_BT)
        logprobs_BT = nn.losses.cross_entropy(logits_BTV, labels_BT)
        loss = (logprobs_BT * pad_mask_BT).sum() / pad_mask_BT.sum()
        return loss


    @partial(mx.compile, inputs=model.state, outputs=model.state)
    def train_step(inputs_BT, labels_BT, grads):
        loss_and_grad = nn.value_and_grad(model, train_forward_pass)
        loss, grads_m = loss_and_grad(model, inputs_BT, labels_BT)
        grads = tree_map(lambda g, gm: (g + gm / cfg_t.grad_acc_steps), grads, grads_m)
        return loss, grads


    grads = tree_map(lambda p: mx.zeros(p.shape), model)
    pbar = tqdm(total=cfg_t.n_steps)
    model.train()

    for step, (inputs_BT, labels_BT) in enumerate(load_train_data()):
        try:
            loss, grads = train_step(inputs_BT, labels_BT, grads)
            mx.eval(loss, grads)

            loss, lr = map(lambda x: x.item(), [loss, optimizer.learning_rate])
            pbar.set_description(f'{loss=:.4f} | {lr=:.2e} | peak_mem={(mx.metal.get_peak_memory()/2**30):.2f}GB')
            pbar.update(1)

            if ((step + 1) % cfg_t.grad_acc_steps == 0) or (step == cfg_t.n_steps - 1):
                optimizer.update(model, grads)
                mx.eval(model.state, optimizer.state)
                grads = tree_map(lambda p: mx.zeros(p.shape), model)
                wandb.log({'loss': loss, 'learning_rate': lr})
        except KeyboardInterrupt:
            break

    pbar.close()
    mx.save_safetensors(cfg_t.ckpt_name, dict(tree_flatten(model)))


if __name__ == '__main__':
    wandb.login(key=os.environ.get('WANDB_KEY', None))  # Set WANDB_MODE="disabled" to disable
    train()
