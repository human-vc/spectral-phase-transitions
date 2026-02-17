import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class TwoLayerReLU(nn.Module):
    def __init__(self, d, m):
        super().__init__()
        self.fc1 = nn.Linear(d, m, bias=False)
        self.fc2 = nn.Linear(m, 1, bias=False)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def compute_hessian_eigenvalues(model, X, y):
    params = list(model.parameters())
    shapes = [p.shape for p in params]
    numels = [p.numel() for p in params]

    def loss_fn(flat_params):
        idx = 0
        w1 = flat_params[idx:idx + numels[0]].view(shapes[0])
        idx += numels[0]
        w2 = flat_params[idx:idx + numels[1]].view(shapes[1])
        h = torch.relu(X @ w1.t())
        pred = h @ w2.t()
        return nn.functional.mse_loss(pred, y)

    flat_current_params = torch.cat([p.detach().view(-1) for p in params])
    H = torch.autograd.functional.hessian(loss_fn, flat_current_params)
    return torch.linalg.eigvalsh(H).detach().cpu().numpy()


def train_to_minimum(n, d, m, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)

    X = torch.randn(n, d).to(device)

    m_teacher = max(1, n // 4)
    teacher = TwoLayerReLU(d, m_teacher).to(device)
    with torch.no_grad():
        nn.init.normal_(teacher.fc1.weight, std=1.0 / np.sqrt(d))
        nn.init.normal_(teacher.fc2.weight, std=1.0 / np.sqrt(m_teacher))
        y = teacher(X)

    model = TwoLayerReLU(d, m).to(device)
    with torch.no_grad():
        nn.init.normal_(model.fc1.weight, std=1.0 / np.sqrt(d))
        nn.init.normal_(model.fc2.weight, std=1.0 / np.sqrt(m))

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for step in range(20000):
        optimizer.zero_grad()
        pred = model(X)
        loss = nn.functional.mse_loss(pred, y)
        loss.backward()
        optimizer.step()

        if loss.item() < 1e-6:
            break

    final_loss = loss.item()

    if final_loss < 1e-3:
        return compute_hessian_eigenvalues(model, X, y), final_loss
    else:
        return None, final_loss


def run_experiment():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu")
    n = 50
    delta = 1.0
    d = int(delta * n)
    gammas = [0.4, 0.6, 0.8, 1.0, 1.4]

    if args.quick:
        gammas = [0.6, 1.0]
        n = 30
        d = 30

    results = {}

    for gamma in gammas:
        m = int(gamma * n)
        eigvals, final_loss = train_to_minimum(n, d, m, 42, device)
        if eigvals is not None:
            results[str(gamma)] = eigvals.tolist()
            print(f"gamma={gamma}, loss={final_loss:.2e}, num_eigs={len(eigvals)}, min_eig={eigvals[0]:.4e}")
        else:
            print(f"gamma={gamma}, loss={final_loss:.2e}, DID NOT CONVERGE - skipping Hessian")

    os.makedirs("experiments/results", exist_ok=True)
    with open("experiments/results/spectral_density.json", "w") as f:
        json.dump(results, f)

    plt.figure(figsize=(12, 7))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(results)))

    gamma_star = 4.0 / (2.0 + 3.0 * delta)

    for i, (gamma_str, evs) in enumerate(sorted(results.items(), key=lambda x: float(x[0]))):
        gamma_val = float(gamma_str)
        label = rf"$\gamma={gamma_val}$"
        if abs(gamma_val - gamma_star) < 0.05:
            label += r" $(\approx \gamma^*)$"
        plt.hist(evs, bins=60, alpha=0.45, label=label, density=True, color=colors[i], edgecolor="none")

    plt.axvline(x=0, color="red", linestyle="--", linewidth=2, label="Zero")
    plt.xlabel("Eigenvalue", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.title(rf"Hessian Spectral Density at Critical Points ($\delta={delta}$, $\gamma^*={gamma_star:.2f}$)", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("experiments/results/spectral_density.pdf", dpi=150)
    plt.close()


if __name__ == "__main__":
    run_experiment()
