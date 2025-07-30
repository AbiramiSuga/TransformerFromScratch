# Attention Is All You Need: Transformer from Scratch

A complete implementation of the Transformer architecture from the groundbreaking paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) using PyTorch. This project is built for educational purposes to understand the inner workings of modern language models.

## 🎯 Overview

This repository contains a from-scratch implementation of the Transformer model, including all core components:
- Multi-Head Self-Attention
- Position-wise Feed-Forward Networks
- Positional Encoding
- Encoder and Decoder Stacks
- Layer Normalization and Residual Connections

## 🏗️ Architecture

The implementation follows the original Transformer architecture:

```
Input → Embedding → Positional Encoding → Encoder Stack → Decoder Stack → Linear → Output
```

### Core Components

1. **Multi-Head Attention**: Allows the model to jointly attend to information from different representation subspaces
2. **Position-wise Feed-Forward Network**: Applies point-wise fully connected layers with GELU activation
3. **Positional Encoding**: Adds positional information using sine and cosine functions
4. **Encoder Layer**: Self-attention + feed-forward with residual connections and layer normalization
5. **Decoder Layer**: Masked self-attention + cross-attention + feed-forward with residual connections

## 🧠 Key Concepts Explained

### Multi-Head Attention
- **Query, Key, Value**: Each token is projected into Q, K, V vectors
- **Multiple Heads**: Different attention heads focus on different aspects (syntax, semantics, etc.)
- **Scaled Dot-Product**: Attention scores computed as `softmax(QK^T/√d_k)V`

### Positional Encoding
- Uses sine and cosine functions to encode position information
- Even positions use sine, odd positions use cosine
- Allows the model to understand token order in sequences

### Masking
- **Padding Mask**: Ignores padding tokens during attention
- **Look-ahead Mask**: Prevents decoder from seeing future tokens during training

## 📚 Educational Features

This implementation includes detailed comments and explanations for:
- Why we use multiple attention heads
- How positional encoding works with sine/cosine functions
- The purpose of layer normalization and residual connections
- Different types of masking in encoder vs decoder
- The role of cross-attention in the decoder

## 🔍 Model Architecture Details

| Component | Description | Key Parameters |
|-----------|-------------|----------------|
| Embedding | Token to vector conversion | `vocab_size × dim_model` |
| Multi-Head Attention | Parallel attention mechanisms | `num_heads`, `dim_head` |
| Feed-Forward | Position-wise MLP | `dim_model → dim_ff → dim_model` |
| Positional Encoding | Sinusoidal position embeddings | `max_seq_length × dim_model` |

## 🎓 Learning Resources

This implementation was inspired by:
- Original paper: ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)
- DataCamp Tutorial: https://www.datacamp.com/tutorial/building-a-transformer-with-py-torch
- Various YouTube videos
- PyTorch documentation and examples

## ⚠️ Important Notes

- **Educational Purpose Only**: This is a learning implementation, not optimized for production
- **Not Trained**: The model architecture is complete but requires training on your dataset
- **Simplified**: Some optimizations from production models are omitted for clarity

## 🚧 Future Improvements

- [ ] Add training loop with sample dataset
- [ ] Implement beam search for inference
- [ ] Add support for different attention variants
- [ ] Include model checkpointing
- [ ] Add visualization tools for attention weights

