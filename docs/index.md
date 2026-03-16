# kinematic-morphospace

**Principal Component Analysis of bird flight morphing.**

kinematic-morphospace is a Python library for analysing the shape changes (morphing) of bird wings and tails during flight using Principal Component Analysis. It provides tools for loading motion-capture data, performing bilateral shape PCA, correcting for whole-body rotations, and comparing shape modes across species.

## Key Features

- **Bilateral shape PCA** — Exploit left-right symmetry to decompose wing and tail postures into interpretable shape modes.
- **Rotation correction** — Remove whole-body pitch, yaw, and roll before extracting shape variation.
- **Cross-species generalisation** — Project shape modes from one species onto another to test universality of morphing strategies.
- **Validation** — Bootstrap, permutation, and null-model tests for statistical robustness.
- **Visualisation** — Publication-ready plots of shape modes, scores, variance explained, and more.

## Getting Started

See the [Quickstart](quickstart.md) guide for installation and basic usage, or browse the [API Reference](api/index.md) for detailed module documentation.
