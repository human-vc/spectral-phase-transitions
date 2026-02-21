import json
import sys
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class TeacherNetwork(nn.Module):
    def __init__(self, d, m_star):
        super().__init__()
        self.fc1 = nn.Linear(d, m_star, bias=False)
        self.fc2 = nn.Linear(m_star, 1, bias=False)
        nn.init.normal_(self.fc1.weight, std=1.0 / np.sqrt(d))
        nn.init.normal_(self.fc2.weight, std=1.0 / np.sqrt(m_star))

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class StudentNetwork(nn.Module):
    def __init__(self, d, m):
        super().__init__()
        self.fc1 = nn.Linear(d, m, bias=False)
        self.fc2 = nn.Linear(m, 1, bias=False)
        nn.init.normal_(self.fc1.weight, std=1.0 / np.sqrt(d))
        nn.init.normal_(self.fc2.weight, std=1.0 / np.sqrt(m))

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def generate_data(n, d, m_star, noise_std, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    X = torch.randn(n, d)
    teacher = TeacherNetwork(d, m_star)
    with torch.no_grad():
        y = teacher(X) + noise_std * torch.randn(n, 1)
    return X, y, teacher


def train_student(X, y, d, m, seed, lr=0.01, epochs=2000, loss_threshold=1e-4):
    torch.manual_seed(seed)
    np.random.seed(seed)
    student = StudentNetwork(d, m)
    optimizer = optim.SGD(student.parameters(), lr=lr)
    criterion = nn.MSELoss()
    n = X.shape[0]

    loss_history = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = student(X)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        loss_val = loss.item()
        loss_history.append(loss_val)
        if loss_val < loss_threshold:
            break

    final_loss = loss_history[-1]
    converged = final_loss < loss_threshold
    return {
        "final_loss": final_loss,
        "converged": bool(converged),
        "epochs_used": len(loss_history),
        "loss_history": loss_history[::max(1, len(loss_history) // 50)],
    }


def compute_gamma_star(delta):
    return 4.0 / (2.0 + 3.0 * delta)


def run_experiment(gamma_values, delta_values, n_base, num_seeds, noise_std,
                   lr, epochs, loss_threshold):
    results = []
    total = len(gamma_values) * len(delta_values) * num_seeds
    count = 0

    for delta in delta_values:
        d = max(1, int(round(delta * n_base)))
        n = n_base
        m_star = max(1, n // 4)
        gamma_star = compute_gamma_star(delta)

        data_seed = 10000
        X, y, _ = generate_data(n, d, m_star, noise_std, data_seed)

        for gamma in gamma_values:
            m = max(1, int(round(gamma * n)))

            for seed in range(num_seeds):
                count += 1
                sys.stdout.write(f"\r  [{count}/{total}] delta={delta:.2f} gamma={gamma:.2f} seed={seed}")
                sys.stdout.flush()

                result = train_student(X, y, d, m, seed, lr, epochs, loss_threshold)
                results.append({
                    "gamma": float(gamma),
                    "delta": float(delta),
                    "gamma_star": float(gamma_star),
                    "n": n,
                    "d": d,
                    "m": m,
                    "m_star": m_star,
                    "seed": seed,
                    **result,
                })

    sys.stdout.write("\n")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-base", type=int, default=50)
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--num-gamma", type=int, default=8)
    parser.add_argument("--num-delta", type=int, default=5)
    parser.add_argument("--gamma-min", type=float, default=0.3)
    parser.add_argument("--gamma-max", type=float, default=3.0)
    parser.add_argument("--delta-min", type=float, default=0.1)
    parser.add_argument("--delta-max", type=float, default=1.0)
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--loss-threshold", type=float, default=1e-3)
    parser.add_argument("--output", type=str, default="results/phase_boundary.json")
    args = parser.parse_args()

    gamma_values = np.linspace(args.gamma_min, args.gamma_max, args.num_gamma).tolist()
    delta_values = np.linspace(args.delta_min, args.delta_max, args.num_delta).tolist()

    print(f"Phase boundary experiment: {args.num_gamma} gamma x {args.num_delta} delta x {args.num_seeds} seeds")
    print(f"  gamma range: [{args.gamma_min}, {args.gamma_max}]")
    print(f"  delta range: [{args.delta_min}, {args.delta_max}]")
    print(f"  n_base={args.n_base}, epochs={args.epochs}, lr={args.lr}")

    results = run_experiment(
        gamma_values, delta_values, args.n_base, args.num_seeds,
        args.noise_std, args.lr, args.epochs, args.loss_threshold,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "config": vars(args),
            "gamma_values": gamma_values,
            "delta_values": delta_values,
            "results": results,
        }, f, indent=2)

    converged_count = sum(1 for r in results if r["converged"])
    print(f"Done. {converged_count}/{len(results)} runs converged. Saved to {args.output}")


if __name__ == "__main__":
    main()
