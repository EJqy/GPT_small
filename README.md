# Full-Stack LLM From Scratch

This project is a full end-to-end implementation of a Large Language Model (LLM) system built from scratch — covering the entire pipeline from raw text to a pretrained Transformer model.

It includes:
- **Tokenizer**: data preprocessing + subword tokenization (BPE)
- **Model**: Transformer decoder for causal language modeling
- **Training**: full pretraining loop with logging, checkpointing, evaluation, and reproducibility utilities

The goal is educational and practical: to understand and build every core component of modern LLM pretraining without relying on high-level training frameworks.

---

## Features

- **BPE tokenizer**: training + encode/decode + special tokens
- **Dataset pipeline**: text ingestion, sharding, batching, collation
- **Transformer implementation**: embeddings, attention, MLP, layer norm, residuals
- **Causal LM objective**: next-token prediction
- **Pretraining system**: optimizer, scheduler, mixed precision (optional), gradient clipping
- **Checkpointing & resume**: save/load model + optimizer state
- **Evaluation**: perplexity on validation set (and extensible hooks)

---

## Project Structure (example)

```text
.
├── tokenizer/            # BPE training + tokenizer implementation
├── model/                # Transformer model code
├── data/                 # dataset building + dataloaders
├── train/                # pretraining scripts, configs, utils
├── checkpoints/          # saved checkpoints (optional)
└── README.md
