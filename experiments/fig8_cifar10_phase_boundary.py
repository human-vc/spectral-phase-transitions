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

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def get_gamma_star(delta):
    """Theoretical phase boundary: gamma* = 4 / (2 + 3*delta)"""
    return 4.0 / (2.0 + 3.0 * delta)

class TwoLayerReLU(nn.Module):
    def __init__(self, d, m):
        super().__init__()
        self.fc1 = nn.Linear(d, m, bias=False)
        self.fc2 = nn.Linear(m, 1, bias=False)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

def load_cifar10_whitened(n, d, seed=0):
    try:
        from torchvision import datasets, transforms
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torchvision", "--quiet"])
        from torchvision import datasets, transforms

    data_dir = Path.home() / ".cache" / "cifar10"
    # Download CIFAR-10
    dataset = datasets.CIFAR10(str(data_dir), train=True, download=True, transform=transforms.ToTensor())

    # Subsample n images
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(dataset), size=n, replace=False)
    
    # Flatten images: (3, 32, 32) -> 3072
    X = torch.stack([dataset[i][0].flatten() for i in indices])
    
    # Labels as floats
    y = torch.tensor([dataset[i][1] for i in indices], dtype=torch.float32).unsqueeze(1)

    # Center data
    mean = X.mean(dim=0)
    X = X - mean

    # PCA to d dimensions
    U, S, Vt = torch.linalg.svd(X, full_matrices=False)
    # Vt[:d] is (d, 3072).
    X_pca = X @ Vt[:d].T

    # Whiten
    cov = (X_pca.T @ X_pca) / n
    L = torch.linalg.cholesky(cov + 1e-6 * torch.eye(d))
    # Solve L Y = X_pca.T => Y = L^{-1} X_pca.T. Then X_white = Y.T
    X_white = torch.linalg.solve_triangular(L, X_pca.T, upper=False).T

    return X_white, y

import gc

def train_one(X, y, m, seed, max_steps=20000, lr=5e-4):
    device = torch.device("cpu")
    n, d = X.shape
    torch.manual_seed(seed)
    
    # Ensure reproducibility on CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = TwoLayerReLU(d, m).to(device)
    # Initialization
    with torch.no_grad():
        nn.init.normal_(model.fc1.weight, std=1.0 / np.sqrt(d))
        nn.init.normal_(model.fc2.weight, std=1.0 / np.sqrt(m))

    X_dev = X.to(device)
    y_dev = y.to(device)

    init_loss = None
    final_loss = None

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for step in range(max_steps):
        pred = model(X_dev)
        loss = F.mse_loss(pred, y_dev)

        if step == 0:
            init_loss = loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss.item() < 1e-8: # Threshold for convergence
            final_loss = loss.item()
            break
    
    if final_loss is None:
        final_loss = loss.item()

    del model, optimizer, X_dev, y_dev
    gc.collect()
    return final_loss

def run_delta(n, d, delta, gammas, num_seeds):
    # Load data inside worker to avoid pickling issues and ensure freshness
    X, y = load_cifar10_whitened(n, d, seed=42)
    print(f"\n--- Starting delta={delta:.2f} (d={d}, n={n}) ---", flush=True)
    
    delta_res = []
    for gamma in gammas:
        m = int(gamma * n)
        if m > 300:
            m = 300
            
        losses = []
        for seed in range(num_seeds):
            fl = train_one(X, y, m, seed)
            losses.append(fl)
            
        gc.collect()
        median_loss = float(np.median(losses))
        delta_res.append({
            "gamma": gamma,
            "median_loss": median_loss,
            "losses": [float(l) for l in losses]
        })
        print(f"delta={delta:.2f}, gamma={gamma:.3f}, m={m}, median_loss={median_loss:.2e}", flush=True)
    
    return delta, delta_res

def run_experiment():
    print("Starting CIFAR-10 phase boundary experiment (Parallel Deltas)...", flush=True)
    
    # Parameters
    n = 200
    deltas = [0.05, 0.10, 0.15]
    dims = [int(delta * n) for delta in deltas] 
    gammas = np.linspace(0.2, 2.0, 10).tolist() 
    num_seeds = 3
    results = {}

    for d, delta in zip(dims, deltas):
        delta, delta_results = run_delta(n, d, delta, gammas, num_seeds)
        results[delta] = delta_results

    # Save Results
    os.makedirs("experiments/results", exist_ok=True)
    with open("experiments/results/cifar10_phase_boundary.json", "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)

    # Plotting
    fig, ax = plt.subplots(figsize=(6, 4.5))

    delta_grid = np.array(deltas)
    gamma_grid = np.array(gammas)
    
    D, G = np.meshgrid(delta_grid, gamma_grid, indexing="ij")
    Z = np.zeros_like(D)

    for i, delta in enumerate(delta_grid):
        data_list = results[delta]
        # Ensure sorted by gamma
        data_list.sort(key=lambda x: x["gamma"])
        # Map back to grid
        for j, g_val in enumerate(gamma_grid):
            # Find entry
            entry = next((item for item in data_list if abs(item["gamma"] - g_val) < 1e-5), None)
            if entry:
                Z[i, j] = np.log10(entry["median_loss"] + 1e-15)

    c = ax.pcolormesh(D, G, Z, cmap="RdYlBu_r", shading="gouraud") 
    fig.colorbar(c, ax=ax, label=r"$\log_{10}(\mathrm{loss})$")

    # Theoretical boundary
    d_theory = np.linspace(0.02, 0.18, 100)
    g_theory = [get_gamma_star(d) for d in d_theory]
    ax.plot(d_theory, g_theory, "k--", linewidth=2, label=r"$\gamma^\star = 4/(2+3\delta)$")

    ax.set_xlabel(r"$\delta = d/n$", fontsize=13)
    ax.set_ylabel(r"$\gamma = m/n$", fontsize=13)
    ax.legend(fontsize=11, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    ax.set_xlim(0.04, 0.16) 
    ax.set_ylim(gammas[0], gammas[-1])
    
    ax.set_title("CIFAR-10 Phase Boundary (n=200)")

    output_path = Path.home() / "projects/spectral-phase-transitions/paper/figures/fig8_cifar10_phase_boundary.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nSaved figure to {output_path}", flush=True)

if __name__ == "__main__":
    run_experiment()
