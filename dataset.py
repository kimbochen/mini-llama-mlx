import logging
from pathlib import Path

import mlx.core as mx
from datasets import load_dataset
from joblib import Memory
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm


logging.basicConfig(level=logging.INFO)
memory = Memory('.data_cache/', verbose=0)


def create_wikitext_dataset(split):
    '''
    Preprocess the WikiText dataset.
    Output:
        text_seqs : List[str] : Processed text sequences.
    '''
    logging.info(f'Loading WikiText {split} dataset ...')
    dataset = load_dataset('wikitext', 'wikitext-103-v1')
    corpus = dataset[split]['text']

    is_title = lambda text: text[:3] == ' = ' and text[-4:] == ' = \n' and text[3].isupper()
    text_seqs = []
    text_seq = corpus[1]  # corpus[0] is an empty string

    for i, text in enumerate(corpus[2:], start=2):
        if (corpus[i-1] == corpus[i-2] == '') and is_title(text):
            text_seqs.append(text_seq)  # Store text sequence when found a new title
            text_seq = text
        else:
            text_seq += text
    else:
        text_seqs.append(text_seq)  # The last text sequence

    return text_seqs


@memory.cache
def prepare_dataset(split, seq_len, pad_token_id):
    '''
    Creates the dataset, tokenize the sequences, splits token sequences to specified length.
    Output:
        token_seqs : mx.array [Number of batches, seq_len]
    '''
    text_seqs = create_wikitext_dataset(split)

    logging.info('Tokenizing dataset ...')
    tokenizer = SentencePieceProcessor(model_file='tokenizer.model')
    blk_size = seq_len + 1
    token_seqs = []

    for text_seq in tqdm(text_seqs):
        token_seq = tokenizer.encode(text_seq, add_bos=True, add_eos=True)
        token_seq = mx.array(token_seq, dtype=mx.int16)
        token_seq = mx.pad(token_seq, [0, blk_size - token_seq.size % blk_size], pad_token_id)
        token_seqs.append(token_seq.reshape(-1, blk_size))

    token_seqs = mx.concatenate(token_seqs, axis=0)
    logging.info(f'Tokenized {token_seqs.shape[0]} batches = {token_seqs.size} tokens.')

    return token_seqs


def config_dataloader(bsz, n_steps, n_epochs, split, seq_len, pad_token_id, **kwargs):
    token_seqs = prepare_dataset(split, seq_len, pad_token_id)
    n_seqs = (bsz * n_steps) // n_epochs
    logging.info(f'Training on {n_seqs} sequences for {n_epochs} epochs.')

    def load_data_batch():
        step_idx = 0
        while True:
            for idx in range(0, n_seqs, bsz):
                bblk = token_seqs[idx:idx+bsz, :]
                yield bblk[:, :-1], bblk[:, 1:]
                step_idx += 1
                if step_idx == n_steps:
                    return

    return load_data_batch


if __name__ == '__main__':
    load_train_data = config_dataloader(256, 10, split='train', seq_len=32, pad_token_id=-1)
    for inputs, labels in load_train_data():
        print(inputs.shape, labels.shape)
