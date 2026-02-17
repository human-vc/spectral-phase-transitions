import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def get_gamma_star(delta):
    return 4.0 / (2.0 + 3.0 * delta)

class TwoLayerReLU(nn.Module):
    def __init__(self, d, m):
        super().__init__()
        self.fc1 = nn.Linear(d, m, bias=False)
        self.fc2 = nn.Linear(m, 1, bias=False)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

def train_one(n, d, m, seed, device):
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
    
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=500)
    
    for step in range(5000):
        optimizer.zero_grad()
        pred = model(X)
        loss = nn.functional.mse_loss(pred, y)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        
        if loss.item() < 1e-4:
            return True
            
    return False

def run_experiment():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = 100
    deltas = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    gammas = np.linspace(0.1, 2.5, 20).tolist()
    num_seeds = 10
    
    if args.quick:
        deltas = [0.5, 1.0]
        gammas = np.linspace(0.5, 1.5, 5).tolist()
        num_seeds = 2
        
    results = {}
    
    for delta in deltas:
        d = int(delta * n)
        results[delta] = []
        for gamma in gammas:
            m = int(gamma * n)
            successes = 0
            for seed in range(num_seeds):
                if train_one(n, d, m, seed, device):
                    successes += 1
            rate = successes / num_seeds
            results[delta].append({"gamma": gamma, "rate": rate})
            print(f"delta={delta}, gamma={gamma:.2f}, rate={rate}")

    os.makedirs("experiments/results", exist_ok=True)
    with open("experiments/results/phase_boundary_empirical.json", "w") as f:
        json.dump(results, f)
        
    plt.figure(figsize=(10, 6))
    
    empirical_deltas = []
    empirical_gammas = []
    
    for delta, res in results.items():
        res.sort(key=lambda x: x["gamma"])
        rates = [r["rate"] for r in res]
        gs = [r["gamma"] for r in res]
        
        transition_gamma = None
        for i in range(len(rates) - 1):
            if rates[i] <= 0.5 and rates[i+1] >= 0.5:
                t = (0.5 - rates[i]) / (rates[i+1] - rates[i] + 1e-9)
                transition_gamma = gs[i] + t * (gs[i+1] - gs[i])
                break
        
        if transition_gamma is not None:
            empirical_deltas.append(delta)
            empirical_gammas.append(transition_gamma)
            
    plt.plot(empirical_deltas, empirical_gammas, 'bo-', label='Empirical (50% success)')
    
    d_grid = np.linspace(min(deltas), max(deltas), 100)
    g_star = [get_gamma_star(d) for d in d_grid]
    plt.plot(d_grid, g_star, 'r--', label='Theoretical gamma*')
    
    plt.xlabel('delta = d/n')
    plt.ylabel('gamma = m/n')
    plt.title('Phase Transition Boundary')
    plt.legend()
    plt.grid(True)
    plt.savefig("experiments/results/phase_boundary_empirical.pdf")

if __name__ == "__main__":
    run_experiment()
