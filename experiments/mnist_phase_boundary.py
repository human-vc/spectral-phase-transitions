import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def get_gamma_star(delta):
    return 4.0 / (2.0 + 3.0 * delta)


class TwoLayerReLU(nn.Module):
    def __init__(self, d, m):
        super().__init__()
        self.fc1 = nn.Linear(d, m, bias=False)
        self.fc2 = nn.Linear(m, 1, bias=False)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


def load_mnist_whitened(n, d, seed=0):
    try:
        from torchvision import datasets, transforms
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torchvision", "--quiet"])
        from torchvision import datasets, transforms

    data_dir = Path.home() / ".cache" / "mnist"
    dataset = datasets.MNIST(str(data_dir), train=True, download=True, transform=transforms.ToTensor())

    rng = np.random.RandomState(seed)
    indices = rng.choice(len(dataset), size=n, replace=False)
    X = torch.stack([dataset[i][0].flatten() for i in indices])
    y = torch.tensor([dataset[i][1] for i in indices], dtype=torch.float32).unsqueeze(1)

    mean = X.mean(dim=0)
    X = X - mean

    U, S, Vt = torch.linalg.svd(X, full_matrices=False)
    X_pca = X @ Vt[:d].T

    cov = (X_pca.T @ X_pca) / n
    L = torch.linalg.cholesky(cov + 1e-6 * torch.eye(d))
    X_white = torch.linalg.solve_triangular(L, X_pca.T, upper=False).T

    return X_white, y


def train_one(X, y, m, seed, max_steps=20000, lr=5e-4):
    n, d = X.shape
    torch.manual_seed(seed)

    model = TwoLayerReLU(d, m)
    with torch.no_grad():
        nn.init.normal_(model.fc1.weight, std=1.0 / np.sqrt(d))
        nn.init.normal_(model.fc2.weight, std=1.0 / np.sqrt(m))

    init_loss = None
    final_loss = None

    for step in range(max_steps):
        pred = model(X)
        loss = F.mse_loss(pred, y)

        if step == 0:
            init_loss = loss.item()

        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            for p in model.parameters():
                p -= lr * p.grad

        if loss.item() < 1e-10:
            final_loss = loss.item()
            break

    if final_loss is None:
        final_loss = loss.item()

    return final_loss


def run_experiment():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    n = 200
    dims = [20, 50, 100, 150]
    gammas = np.linspace(0.2, 3.0, 25).tolist()
    num_seeds = 5

    if args.quick:
        dims = [30, 50, 80]
        gammas = np.linspace(0.3, 2.5, 12).tolist()
        num_seeds = 3

    results = {}

    for d in dims:
        delta = d / n
        print(f"\n--- delta={delta:.2f} (d={d}, n={n}) ---", flush=True)
        X, y = load_mnist_whitened(n, d, seed=42)

        results[delta] = []
        for gamma in gammas:
            m = max(1, int(gamma * n))
            losses = []
            for seed in range(num_seeds):
                fl = train_one(X, y, m, seed)
                losses.append(fl)
            median_loss = float(np.median(losses))
            results[delta].append({
                "gamma": gamma,
                "median_loss": median_loss,
                "losses": [float(l) for l in losses],
            })
            print(f"delta={delta:.2f}, gamma={gamma:.3f}, m={m}, median_loss={median_loss:.2e}", flush=True)

    os.makedirs("experiments/results", exist_ok=True)
    with open("experiments/results/mnist_phase_boundary.json", "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)

    fig, ax = plt.subplots(figsize=(6, 4.5))

    delta_grid = np.array(sorted(results.keys()))
    gamma_grid = np.array(gammas)
    D, G = np.meshgrid(delta_grid, gamma_grid, indexing="ij")
    Z = np.zeros_like(D)

    for i, delta in enumerate(delta_grid):
        for j, entry in enumerate(results[delta]):
            Z[i, j] = np.log10(entry["median_loss"] + 1e-15)

    c = ax.pcolormesh(D, G, Z, cmap="RdYlBu_r", shading="gouraud")
    fig.colorbar(c, ax=ax, label=r"$\log_{10}(\mathrm{loss})$")

    d_theory = np.linspace(0.05, max(delta_grid) + 0.1, 200)
    g_theory = [get_gamma_star(d) for d in d_theory]
    ax.plot(d_theory, g_theory, "k--", linewidth=2, label=r"$\gamma^\star = 4/(2+3\delta)$")

    ax.set_xlabel(r"$\delta = d/n$", fontsize=13)
    ax.set_ylabel(r"$\gamma = m/n$", fontsize=13)
    ax.legend(fontsize=11, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(delta_grid[0], delta_grid[-1])
    ax.set_ylim(gammas[0], gammas[-1])
    fig.tight_layout()
    fig.savefig("experiments/results/mnist_phase_boundary.pdf", dpi=200, bbox_inches="tight")
    plt.close()
    print("\nSaved mnist_phase_boundary.pdf", flush=True)


if __name__ == "__main__":
    run_experiment()
