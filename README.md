# Spectral Phase Transitions in the Loss Landscape of Finite-Width Neural Networks

**Research Note — February 2026**

## Summary

We study the critical-point structure of the empirical risk landscape for two-layer ReLU neural networks trained on $n$ data points in $\mathbb{R}^d$ with $m$ hidden neurons. Our main result establishes a **sharp phase transition** in the Hessian spectrum at critical points:

- **Supercritical regime** ($\gamma = m/n > \gamma_\star$): All local minima are global with probability $1 - e^{-\Omega(n)}$.
- **Subcritical regime** ($\gamma < \gamma_\star$): Exponentially many spurious local minima exist.

### The Critical Ratio

For isotropic data ($\Sigma = I_d$) with $\delta = d/n$:

$$\gamma_\star(\delta) = \frac{4}{2 + 3\delta}$$

For $\delta = 1$ (i.e., $d = n$): $\gamma_\star = 4/5$, meaning $m \geq \lceil 4n/5 \rceil$ hidden neurons suffice.

### Universal Scaling Law

At the transition, the spectral gap scales as $|\gamma - \gamma_\star|^{1/2}$, yielding a universal critical exponent $\beta = 1/2$.

## Repository Structure

```
paper/          LaTeX source and compiled PDF
experiments/    Numerical experiments (forthcoming)
```

## Building the Paper

```bash
cd paper
pdflatex main.tex
pdflatex main.tex  # run twice for cross-references
```

## Key Techniques

- **Spectral decoupling**: Decomposition of the Hessian at critical points into data and weight contributions
- **Kac–Rice formula**: Counting critical points via the expected number formula
- **Random matrix theory**: Marchenko–Pastur law and free convolution for the limiting spectrum
- **Tracy–Widom fluctuations**: Finite-$n$ corrections to the spectral edge

## License

MIT License. Copyright (c) 2026 Jacob Crainic.
