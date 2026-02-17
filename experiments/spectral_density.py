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
        w1 = flat_params[idx:idx+numels[0]].view(shapes[0])
        idx += numels[0]
        w2 = flat_params[idx:idx+numels[1]].view(shapes[1])
        
        h = torch.relu(X @ w1.t())
        pred = h @ w2.t()
        return nn.functional.mse_loss(pred, y)
    
    flat_current_params = torch.cat([p.view(-1) for p in params])
    H = torch.autograd.functional.hessian(loss_fn, flat_current_params)
    return torch.linalg.eigvalsh(H).detach().cpu().numpy()

def train_to_critical_point(n, d, m, seed, device, quick=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    X = torch.randn(n, d).to(device)
    
    m_teacher = max(1, n // 4)
    teacher = TwoLayerReLU(d, m_teacher).to(device)
    with torch.no_grad():
        nn.init.normal_(teacher.fc1.weight, std=1.0/np.sqrt(d))
        nn.init.normal_(teacher.fc2.weight, std=1.0/np.sqrt(m_teacher))
        y = teacher(X)
    
    model = TwoLayerReLU(d, m).to(device)
    with torch.no_grad():
        nn.init.normal_(model.fc1.weight, std=1.0/np.sqrt(d))
        nn.init.normal_(model.fc2.weight, std=1.0/np.sqrt(m))
        
    optimizer = optim.SGD(model.parameters(), lr=1e-4)
    
    max_steps = 1000 if quick else 20000
    tol = 1e-6
    
    for step in range(max_steps):
        optimizer.zero_grad()
        pred = model(X)
        loss = nn.functional.mse_loss(pred, y)
        loss.backward()
        
        grad_norm = torch.cat([p.grad.view(-1) for p in model.parameters()]).norm()
        if grad_norm < tol:
            break
            
        optimizer.step()
        
    return compute_hessian_eigenvalues(model, X, y)

def run_experiment():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    
    device = torch.device("cpu")
    n = 100
    d = 100
    gammas = [0.5, 0.7, 0.8, 0.9, 1.2]
    
    if args.quick:
        gammas = [0.5, 1.2]
        n = 30
        d = 30
        
    results = {}
    
    for gamma in gammas:
        m = int(gamma * n)
        eigvals = train_to_critical_point(n, d, m, 42, device, args.quick)
        results[str(gamma)] = eigvals.tolist()
        
    os.makedirs("experiments/results", exist_ok=True)
    with open("experiments/results/spectral_density.json", "w") as f:
        json.dump(results, f)
        
    plt.figure(figsize=(10, 6))
    for gamma, evs in results.items():
        plt.hist(evs, bins=50, alpha=0.5, label=f"gamma={gamma}", density=True)
        
    plt.axvline(x=0, color='k', linestyle='--')
    plt.title('Hessian Eigenvalue Density')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig("experiments/results/spectral_density.pdf")

if __name__ == "__main__":
    run_experiment()
