from pathlib import Path

import mlx.core as mx
from datasets import load_dataset
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm

DATASET_DIR = 'wikitext_data/train/'


def create_wikitext_dataset():
    print(f'Creating WikiText dataset at {DATASET_DIR}...')
    dataset = load_dataset('wikitext', 'wikitext-103-v1')

    examples = []
    example = train_text[1]  # train_text[0] is emtpy
    for text in train_text[2:]:
        if (text[:3] == ' = ' and text[-4:] == ' = \n') and text[3].isupper():
            examples.append(example)
            example = text
        else:
            example += text

    sp_model = SentencePieceProcessor(model_file='tokenizer.model')
    def encode(text):
        tokens = sp_model.encode(text, add_bos=True, add_eos=True)
        token_arr = mx.array(tokens, dtype=mx.uint16)
        return token_arr
    train_tokens = [encode(example) for example in tqdm(examples)]

    chunk_size = 1000
    for chunk_idx, idx in enumerate(range(0, len(train_tokens), chunk_size)):
        mx.savez(f'{DATASET_DIR}/example_chunk{chunk_idx:02d}', *train_tokens[idx:idx+chunk_size])


def config_dataloader(bsz, seq_len, pad_token_id, n_steps, **kwargs):
    train_data_dir = Path(DATASET_DIR)
    assert train_data_dir.exists(), f'Invalid path {train_data_dir}; pwd: {Path("./").absolute()}'
    train_examples = []
    for ex_path in sorted(train_data_dir.glob('*.npz')):
        train_examples.extend(mx.load(str(ex_path)).values())

    blk_size = seq_len + 1
    pad_example = lambda ex: mx.pad(ex, [0, blk_size - ex.size % blk_size], pad_token_id)
    train_examples = mx.concatenate([*map(pad_example, train_examples)], axis=0)

    bblk_size = bsz * blk_size  # Batch block size
    print(f'Training dataset: {train_examples.size:.3e} tokens')

    def load_data():
        for i in range(n_steps):
            bblk = train_examples[i*bblk_size:(i+1)*bblk_size].reshape([bsz, blk_size])
            yield bblk[:, :-1], bblk[:, 1:]

    return load_data()


if __name__ == '__main__':
    tokenizer = SentencePieceProcessor(model_file='tokenizer.model')
    dataloader = config_dataloader(2, 32, -1, 10)
