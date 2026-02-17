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


def get_gamma_star(delta):
    return 4.0 / (2.0 + 3.0 * delta)


class TwoLayerReLU(nn.Module):
    def __init__(self, d, m):
        super().__init__()
        self.fc1 = nn.Linear(d, m, bias=False)
        self.fc2 = nn.Linear(m, 1, bias=False)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


def train_one(n, d, m, seed, device, max_steps=20000, lr=5e-4):
    torch.manual_seed(seed)
    np.random.seed(seed)

    X = torch.randn(n, d, device=device)
    m_teacher = max(2, n // 2)
    teacher = TwoLayerReLU(d, m_teacher).to(device)
    with torch.no_grad():
        nn.init.normal_(teacher.fc1.weight, std=2.0 / np.sqrt(d))
        nn.init.normal_(teacher.fc2.weight, std=2.0 / np.sqrt(m_teacher))
        y = teacher(X) + 0.001 * torch.randn(n, 1, device=device)

    model = TwoLayerReLU(d, m).to(device)
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

    relative_reduction = final_loss / (init_loss + 1e-15)
    return final_loss, relative_reduction


def run_experiment():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = 100
    deltas = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    gammas = np.linspace(0.1, 3.0, 30).tolist()
    num_seeds = 10

    if args.quick:
        deltas = [0.25, 0.5, 1.0, 1.5]
        gammas = np.linspace(0.2, 2.5, 12).tolist()
        num_seeds = 3

    results = {}

    for delta in deltas:
        d = max(1, int(delta * n))
        results[delta] = []
        for gamma in gammas:
            m = max(1, int(gamma * n))
            losses = []
            reductions = []
            for seed in range(num_seeds):
                fl, rr = train_one(n, d, m, seed, device)
                losses.append(fl)
                reductions.append(rr)
            median_loss = float(np.median(losses))
            median_reduction = float(np.median(reductions))
            results[delta].append({
                "gamma": gamma,
                "median_loss": median_loss,
                "median_reduction": median_reduction,
                "losses": [float(l) for l in losses],
            })
            print(f"delta={delta}, gamma={gamma:.3f}, median_loss={median_loss:.2e}, reduction={median_reduction:.4f}", flush=True)

    os.makedirs("experiments/results", exist_ok=True)
    with open("experiments/results/phase_boundary_empirical.json", "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)

    fig, ax = plt.subplots(figsize=(6, 4.5))

    delta_grid = np.array(deltas)
    gamma_grid = np.array(gammas)
    D, G = np.meshgrid(delta_grid, gamma_grid, indexing="ij")
    Z = np.zeros_like(D)

    for i, delta in enumerate(deltas):
        for j, entry in enumerate(results[delta]):
            Z[i, j] = np.log10(entry["median_loss"] + 1e-15)

    c = ax.pcolormesh(D, G, Z, cmap="RdYlBu_r", shading="gouraud")
    cb = fig.colorbar(c, ax=ax, label=r"$\log_{10}(\mathrm{loss})$")

    d_theory = np.linspace(0.05, 2.5, 200)
    g_theory = [get_gamma_star(d) for d in d_theory]
    ax.plot(d_theory, g_theory, "k--", linewidth=2, label=r"$\gamma^\star = 4/(2+3\delta)$")

    ax.set_xlabel(r"$\delta = d/n$", fontsize=13)
    ax.set_ylabel(r"$\gamma = m/n$", fontsize=13)
    ax.legend(fontsize=11, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(deltas[0], deltas[-1])
    ax.set_ylim(gammas[0], gammas[-1])
    fig.tight_layout()
    fig.savefig("experiments/results/phase_boundary_empirical.pdf", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved phase_boundary_empirical.pdf")


if __name__ == "__main__":
    run_experiment()
