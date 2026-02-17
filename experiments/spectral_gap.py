import json
import sys
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd.functional import hessian


class TwoLayerReLU(nn.Module):
    def __init__(self, d, m):
        super().__init__()
        self.d = d
        self.m = m
        self.fc1 = nn.Linear(d, m, bias=False)
        self.fc2 = nn.Linear(m, 1, bias=False)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class TeacherNetwork(nn.Module):
    def __init__(self, d, m_star):
        super().__init__()
        self.fc1 = nn.Linear(d, m_star, bias=False)
        self.fc2 = nn.Linear(m_star, 1, bias=False)
        nn.init.normal_(self.fc1.weight, std=1.0 / np.sqrt(d))
        nn.init.normal_(self.fc2.weight, std=1.0 / np.sqrt(m_star))

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def generate_data(n, d, m_star, noise_std, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    X = torch.randn(n, d)
    teacher = TeacherNetwork(d, m_star)
    with torch.no_grad():
        y = teacher(X) + noise_std * torch.randn(n, 1)
    return X, y


def make_loss_fn(model, X, y):
    def loss_fn(*params):
        idx = 0
        for p in model.parameters():
            numel = p.numel()
            p.data = params[idx].reshape(p.shape)
            idx += 1
        pred = model(X)
        return nn.functional.mse_loss(pred, y)
    return loss_fn


def compute_hessian_eigenvalues(model, X, y, num_eigenvalues=5):
    param_list = tuple(p.contiguous().view(-1) for p in model.parameters())
    flat_params = torch.cat(param_list)
    total_params = flat_params.numel()

    if total_params > 2000:
        return approximate_hessian_eigenvalues(model, X, y, num_eigenvalues)

    def flat_loss(flat_p):
        idx = 0
        params_reconstructed = []
        for p in model.parameters():
            numel = p.numel()
            params_reconstructed.append(flat_p[idx:idx + numel].reshape(p.shape))
            idx += numel

        h = torch.relu(X @ params_reconstructed[0].t())
        pred = h @ params_reconstructed[1].t()
        return nn.functional.mse_loss(pred, y)

    H = torch.autograd.functional.hessian(flat_loss, flat_params)
    H_np = H.detach().numpy()
    H_sym = 0.5 * (H_np + H_np.T)
    eigenvalues = np.linalg.eigvalsh(H_sym)
    return eigenvalues


def approximate_hessian_eigenvalues(model, X, y, num_eigenvalues=5, num_iterations=100):
    params = [p for p in model.parameters()]
    flat_params = torch.cat([p.contiguous().view(-1) for p in params])
    total_params = flat_params.numel()
    k = min(num_eigenvalues, total_params)

    def hvp(vec):
        model.zero_grad()
        pred = model(X)
        loss = nn.functional.mse_loss(pred, y)
        grads = torch.autograd.grad(loss, params, create_graph=True)
        flat_grad = torch.cat([g.contiguous().view(-1) for g in grads])
        grad_vec_product = torch.sum(flat_grad * vec)
        hvp_grads = torch.autograd.grad(grad_vec_product, params, retain_graph=True)
        return torch.cat([g.contiguous().view(-1) for g in hvp_grads]).detach()

    V = torch.randn(total_params, k)
    V, _ = torch.linalg.qr(V)

    for _ in range(num_iterations):
        AV = torch.zeros_like(V)
        for j in range(k):
            AV[:, j] = hvp(V[:, j])
        V, _ = torch.linalg.qr(AV)

    T = torch.zeros(k, k)
    for j in range(k):
        Av = hvp(V[:, j])
        for i in range(k):
            T[i, j] = torch.dot(V[:, i], Av)

    eigenvalues = torch.linalg.eigvalsh(T).numpy()
    return eigenvalues


def train_to_critical_point(X, y, d, m, seed, lr=0.01, epochs=3000):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = TwoLayerReLU(d, m)
    nn.init.normal_(model.fc1.weight, std=1.0 / np.sqrt(d))
    nn.init.normal_(model.fc2.weight, std=1.0 / np.sqrt(m))
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X)
        loss = nn.functional.mse_loss(pred, y)
        loss.backward()
        grad_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters()) ** 0.5
        optimizer.step()
        if grad_norm < 1e-5:
            break

    with torch.no_grad():
        final_loss = nn.functional.mse_loss(model(X), y).item()

    return model, final_loss


def compute_gamma_star(delta):
    return 4.0 / (2.0 + 3.0 * delta)


def run_spectral_experiment(delta_values, gamma_offsets, n_base, num_seeds,
                            noise_std, lr, epochs):
    results = []
    total = len(delta_values) * len(gamma_offsets) * num_seeds
    count = 0

    for delta in delta_values:
        d = max(1, int(round(delta * n_base)))
        n = n_base
        m_star = max(1, n // 4)
        gs = compute_gamma_star(delta)

        data_seed = 10000
        X, y = generate_data(n, d, m_star, noise_std, data_seed)

        for offset in gamma_offsets:
            gamma = gs + offset
            if gamma <= 0:
                continue
            m = max(1, int(round(gamma * n)))

            for seed in range(num_seeds):
                count += 1
                sys.stdout.write(f"\r  [{count}/{total}] delta={delta:.2f} gamma={gamma:.2f} offset={offset:+.2f} seed={seed}")
                sys.stdout.flush()

                model, final_loss = train_to_critical_point(X, y, d, m, seed, lr, epochs)
                eigenvalues = compute_hessian_eigenvalues(model, X, y)
                min_eigenvalue = float(eigenvalues[0])
                spectral_gap = float(eigenvalues[1] - eigenvalues[0]) if len(eigenvalues) > 1 else 0.0
                is_saddle = min_eigenvalue < -1e-6
                is_local_min = min_eigenvalue > -1e-6

                results.append({
                    "gamma": float(gamma),
                    "delta": float(delta),
                    "gamma_star": float(gs),
                    "gamma_offset": float(offset),
                    "n": n,
                    "d": d,
                    "m": m,
                    "seed": seed,
                    "final_loss": final_loss,
                    "min_eigenvalue": min_eigenvalue,
                    "spectral_gap": spectral_gap,
                    "is_saddle": bool(is_saddle),
                    "is_local_min": bool(is_local_min),
                    "top_eigenvalues": [float(e) for e in eigenvalues[:5]],
                })

    sys.stdout.write("\n")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-base", type=int, default=30)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--num-delta", type=int, default=4)
    parser.add_argument("--delta-min", type=float, default=0.2)
    parser.add_argument("--delta-max", type=float, default=0.8)
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--output", type=str, default="results/spectral_gap.json")
    args = parser.parse_args()

    delta_values = np.linspace(args.delta_min, args.delta_max, args.num_delta).tolist()
    gamma_offsets = [-0.8, -0.5, -0.3, -0.15, -0.05, 0.0, 0.05, 0.15, 0.3, 0.5, 0.8]

    print(f"Spectral gap experiment: {args.num_delta} delta x {len(gamma_offsets)} offsets x {args.num_seeds} seeds")
    print(f"  delta range: [{args.delta_min}, {args.delta_max}]")
    print(f"  gamma offsets from gamma*: {gamma_offsets}")

    results = run_spectral_experiment(
        delta_values, gamma_offsets, args.n_base, args.num_seeds,
        args.noise_std, args.lr, args.epochs,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "config": vars(args),
            "delta_values": delta_values,
            "gamma_offsets": gamma_offsets,
            "results": results,
        }, f, indent=2)

    print(f"Done. {len(results)} spectral measurements saved to {args.output}")


if __name__ == "__main__":
    main()
