# Full-Stack LLM From Scratch

This project is aimed to a full end-to-end implementation of a Large Language Model (LLM) system built from scratch — covering the entire pipeline from raw text to a ready-to-use model.

It includes:
- **Tokenizer**: data preprocessing + subword tokenization (BPE)
- **Model**: Transformer architecture for causal language modeling
- **Training**: full pretraining loop with logging, checkpointing, evaluation, and reproducibility utilities

The goal is educational and practical: to understand and build every core component of modern LLM pretraining without relying on high-level frameworks.

---

## Features

- **BPE tokenizer**: training + encode/decode + special tokens
- **Transformer implementation**: embeddings, attention, MLP, layer norm, residuals
- **Mixture of Experts**: optional
- **Causal LM objective**: next-token prediction
- **Pretraining system**: optimizer, scheduler, gradient clipping
- **Checkpointing & resume**: save/load model + optimizer state

---

## Project Structure

```text
.
├── tokenizer/            # BPE training + tokenizer implementation
├── model/                # Transformer model code
└── README.md
