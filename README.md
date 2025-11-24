# Neurips2025-CMDX-Tutorial

Positional Encoding: Past, Present, and Future
Unified Implementations of Sinusoidal · Learned Absolute · Axial · Shaw · TENER · Distance-Aware · ALiBi · Huang

This repository provides a research-oriented, pedagogically faithful collection of positional encoding (PE) methods used across Transformer architectures. Each implementation is written with an emphasis on mathematical clarity, transparent tensor operations, and conceptual fidelity to the original formulation.

The goal is not to offer production-level efficiency, but to serve as a clear reference for students and researchers studying how positional information is represented, transformed, and integrated within self-attention.

🚀 Motivation

Self-attention is inherently permutation-invariant: without additional structure, it cannot detect or reason about order. Positional encodings provide the structured inductive bias needed for Transformers to interpret sequential, spatial, and relational data.

However, positional encoding is no longer a single idea—it's an entire design space:

Absolute methods encode where a token occurs.

Relative methods encode how tokens relate to one another.

Bias-based and attention-modifying methods reshape attention scores directly.

2D/axial methods extend these ideas to images and grids.

Hybrid and kernel-based approaches explicitly model distance as a continuous function.

This repository brings these methods together under a unified conceptual lens, enabling direct comparison, experimentation, and deeper understanding of their inductive biases.

📘 Content Overview

The repository includes faithful, readable implementations of eight major families of positional encoding:

1. Sinusoidal (Absolute, Additive)

The classical formulation using fixed-frequency bands.
Highlights how absolute position is injected into token embeddings and how frequency scales relate to locality and global structure.

2. Learned Absolute Positional Embeddings

A trainable extension of absolute encoding for tasks where learned geometry offers better task specialization.

3. Axial Positional Encoding (2D / Multi-Dimensional)

Designed for images and grid-like inputs.
Encodes position factorized across each axis, reducing parameter cost and extending absolute encodings beyond 1D sequences.

4. Shaw-Style Relative Position Encoding

Injects relative displacement directly into attention scores.
Captures translation-invariant relationships and strengthens the model’s ability to represent local ordering patterns.

5. TENER-Style Relative Encoding

An architecture supporting directional bias and multi-head specialization over relative displacements.
Designed for labeling and token-level modeling tasks.

6. Huang-Style Attention-Modified Encoding

A series of four modified attention mechanisms derived from Shaw's work. 

7. Distance-Aware (DA) Encoding

A sinusoidal-based approach where attention weights are modulated by continuous functions of distance.

8. ALiBi (Attention Linear Bias)

A monotonic bias term ensuring strong extrapolation and inductive alignment for very long contexts.
Lightweight, parameter-efficient, and widely used in long-context models.


📚 Educational Philosophy

This repository is intentionally built for learning and conceptual clarity:

Clear, explicit tensor operations

Direct mapping from equations to code

Minimal abstraction, maximal transparency

Interchangeable modules for comparative evaluation

Each method reflects its original conceptual purpose, helping researchers understand why each encoding behaves the way it does.

⚠️ Disclaimer (Research-Grade Educational Reference)

This repository prioritizes clarity and faithfulness over optimization:

Disclaimer:
This implementation is provided for educational and research reference.
Its primary purpose is to illustrate the mechanics and mathematical structure of each positional encoding method in a transparent, developer-friendly manner.
The design intentionally favors readability and explicitness over computational efficiency or engineering optimizations.

While correct and faithful to the underlying ideas, these implementations are not intended as production-grade components.
Industrial systems may use fused kernels, compressed representations, or algorithmic shortcuts that are intentionally omitted here to preserve conceptual clarity.

Author: Christopher Curtis
Email: curtis.ch@northeastern.edu
