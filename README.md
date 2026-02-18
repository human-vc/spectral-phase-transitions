# Spectral Phase Transitions in the Loss Landscape of Neural Networks

This repository investigates the critical-point structure of the empirical risk landscape for two-layer ReLU neural networks. We identify a sharp topological phase transition in the Hessian spectrum, determined by a critical width-to-sample ratio $\gamma_\star$.

## Key Findings

- **Topological Phase Transition:** Above a critical ratio $\gamma_\star$, the loss landscape is benign (all local minima are global). Below it, spurious local minima proliferate exponentially.
- **Exact Threshold:** For isotropic data, the critical ratio is given by $\gamma_\star(\delta) = 2(1-2\delta)/(1-\delta-\delta^2)$, or approximately $4/(2+3\delta)$.
- **Unconditional Proof:** We establish these results unconditionally using a deterministic equivalent for the gated Hessian resolvent (via anisotropic local laws).
- **Optimization Gap:** While the landscape topology suggests optimization should fail below $\gamma_\star$, gradient-based optimizers succeed well into the subcritical regime, highlighting a fundamental gap between landscape geometry and optimization dynamics.

## Repository Structure

- `paper/`: LaTeX source and compiled PDF of the main paper.
- `experiments/`: Code for reproducing the phase transition plots, Hessian spectral density analysis, and training dynamics experiments (MNIST, CIFAR-10).

## Building the Paper

```bash
cd paper
pdflatex main.tex
pdflatex main.tex  
```

## License

MIT License. Copyright (c) 2026 Jacob Crainic.
