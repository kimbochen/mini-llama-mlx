# Mini-LLaMA MLX

A simple implementation of LLaMA 2 that you can run experiments with on your MacBook.


## Setup

- Download the LLaMA 2 tokenizer from [Meta's website](https://llama.meta.com/llama-downloads/)
- Install the required packages by `pip install -r requirements.txt`
- To train the model, run `python train.py`. The training configuration is in the `TrainerConfig` class.
- To use the model, run `python generate.py <checkpoint_path> <prompt>`.


## Why Another Implementation

I decided to write another implementation for a couple reasons:

- Try out MLX: MLX is a new framework with clean APIs and has the good features of JAX and PyTorch. I want to try it out.
- Educational: I want to create an example with more modern architectures and techniques. I will also explain the code in detail.
- Lightweight: I want to create something that enables people with limited compute to experiment with language models.
- Simple and hackable: I limit support to reduce code complexity, but the code being simple makes it easy to hack.


## Dataset

Inspired by [LLM-baselines](https://github.com/epfml/llm-baselines),
I chose [WikiText](https://huggingface.co/datasets/wikitext) as the dataset.  
WikiText is small (~100M tokens) but large enough to be interesting.

### Data Preprocessing

WikiText dataset is a collection of articles, and the one Hugging Face Dataset hosts splits it into sentences.  
To avoid the content of two articles showing up in one sequence,
I decided to concatenate sentences in the same article and split batches by articles.
This turned out to be surprisingly difficult to get it right.

The title of each article is formatted as `= Title =`, and sections `= = section title = =` or `= = = subsection title = = =`.  
Initially I applied a heuristic that classifies a sequence as a title if it starts with exactly one `=`,
but turns out game stats in sports are formatted in the same way as titles are: `= Win ; D =`.  
I tried to classify sentences with `;` as non-titles, but I found that there are some titles that has `;`.  
Even after removing game stats, some sentences that start with math equations are still misclassified, e.g. `= A * B + C =`.

I learned that **creating a high quality dataset is hard, especially when we try to scale up.**  
Data correctness cannot be overlooked because data is the upper bound of a model's performance.  
I ended up erring on the cautious side by splitting everything that looks like a title and drop articles that are too short.


### Sequence and Batching


Because the data size is small, I decided to tokenize everything beforehand to make training faster.

## Model


### Attention


### Feed Forward Network


### Loss Function


## Training


### Weight Initialization


### Learning Rate Scheduling


### Evaluation Metric


## Generation


### Sampling


## Future Goals
