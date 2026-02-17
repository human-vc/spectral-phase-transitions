import argparse
import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def compute_gamma_star(delta):
    return 4.0 / (2.0 + 3.0 * delta)

class TwoLayerReLU(nn.Module):
    def __init__(self, d, m):
        super().__init__()
        self.d = d
        self.m = m
        self.fc1 = nn.Linear(d, m, bias=False)
        self.fc2 = nn.Linear(m, 1, bias=False)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

def compute_full_hessian(model, X, y):
    params = list(model.parameters())
    shapes = [p.shape for p in params]
    numels = [p.numel() for p in params]
    
    def loss_fn(flat_params):
        # Reconstruct params from flat vector
        idx = 0
        w1 = flat_params[idx:idx+numels[0]].view(shapes[0])
        idx += numels[0]
        w2 = flat_params[idx:idx+numels[1]].view(shapes[1])
        
        # Manual forward pass
        h = torch.relu(X @ w1.t())
        pred = h @ w2.t()
        return nn.functional.mse_loss(pred, y)
    
    flat_current_params = torch.cat([p.view(-1) for p in params])
    H = torch.autograd.functional.hessian(loss_fn, flat_current_params)
    return H

def train_to_critical_point(model, X, y, lr=1e-4, epochs=10000, tol=1e-6):
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        
        grad_norm = torch.cat([p.grad.view(-1) for p in model.parameters()]).norm()
        if grad_norm < tol:
            break
            
        optimizer.step()
    
    return loss.item()

def run_experiment(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    results = []
    
    delta_grid = np.linspace(args.delta_min, args.delta_max, args.num_delta)
    if args.quick:
        delta_grid = delta_grid[:2]
        args.num_seeds = 1
        args.epochs = 1000
        
    gamma_offsets = [-0.5, -0.2, 0.0, 0.2, 0.5]
    
    for delta in delta_grid:
        d = int(delta * args.n)
        gamma_star = compute_gamma_star(delta)
        
        for offset in gamma_offsets:
            gamma = gamma_star + offset
            if gamma <= 0: continue
            
            m = int(gamma * args.n)
            
            for seed in range(args.num_seeds):
                # Teacher
                m_teacher = max(1, args.n // 4)
                teacher = TwoLayerReLU(d, m_teacher)
                with torch.no_grad():
                    teacher.fc1.weight.normal_(0, 1/np.sqrt(d))
                    teacher.fc2.weight.normal_(0, 1/np.sqrt(m_teacher))
                
                X = torch.randn(args.n, d)
                with torch.no_grad():
                    y = teacher(X) + args.noise * torch.randn(args.n, 1)
                
                model = TwoLayerReLU(d, m)
                with torch.no_grad():
                    model.fc1.weight.normal_(0, 1/np.sqrt(d))
                    model.fc2.weight.normal_(0, 1/np.sqrt(m))
                
                final_loss = train_to_critical_point(model, X, y, lr=args.lr, epochs=args.epochs)
                
                H = compute_full_hessian(model, X, y)
                eigvals = torch.linalg.eigvalsh(H)
                min_eig = eigvals[0].item()
                
                results.append({
                    "delta": float(delta),
                    "gamma": float(gamma),
                    "gamma_star": float(gamma_star),
                    "gamma_offset": float(offset),
                    "min_eig": min_eig,
                    "final_loss": final_loss,
                    "seed": seed
                })
                
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--delta_min", type=float, default=0.5)
    parser.add_argument("--delta_max", type=float, default=2.0)
    parser.add_argument("--num_delta", type=int, default=4)
    parser.add_argument("--num_seeds", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50000)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=str, default="experiments/results/spectral_gap.json")
    args = parser.parse_args()
    
    results = run_experiment(args)
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f)
