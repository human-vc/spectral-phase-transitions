"""
Experiment 2: Test spurious minima on real data (MNIST/CIFAR).

Tests whether the phenomenon appears with real-world data,
or if the structure of real images makes the landscape different.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import json
import os


class TwoLayerReLU(nn.Module):
    def __init__(self, d, m, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(d, m, bias=False)
        self.fc2 = nn.Linear(m, num_classes, bias=False)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


def compute_gamma_star(delta):
    return 4.0 / (2.0 + 3.0 * delta)


def train_model(model, train_loader, epochs=100, lr=0.01):
    """Train model and return final loss."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
    
    return model


def evaluate_loss(model, data_loader):
    """Compute average loss on data."""
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    count = 0
    
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            output = model(batch_x)
            loss = criterion(output, batch_y)
            total_loss += loss.item() * len(batch_y)
            count += len(batch_y)
    
    return total_loss / count


def find_global_minimum(train_loader, d, m, num_classes, num_seeds=20):
    """Find the minimum loss across multiple random initializations."""
    best_loss = float('inf')
    best_state = None
    
    for seed in range(num_seeds):
        torch.manual_seed(seed)
        
        model = TwoLayerReLU(d, m, num_classes)
        with torch.no_grad():
            model.fc1.weight.normal_(0, 1 / np.sqrt(d))
            model.fc2.weight.normal_(0, 1 / np.sqrt(m))
        
        model = train_model(model, train_loader, epochs=50)
        loss = evaluate_loss(model, train_loader)
        
        if loss < best_loss:
            best_loss = loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    return best_loss, best_state


def run_real_data_experiment(dataset_name="mnist"):
    """Test for spurious minima on real data."""
    
    # Load data
    if dataset_name == "mnist":
        from torchvision import datasets, transforms
        transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
        full_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    else:
        from torchvision import datasets, transforms
        transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
        full_dataset = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
    
    # Use subset for speed
    n = 500  # samples
    indices = list(range(n))
    subset = Subset(full_dataset, indices)
    train_loader = DataLoader(subset, batch_size=50, shuffle=True)
    
    # Get input dimension
    d = 784 if dataset_name == "mnist" else 3072
    num_classes = 10
    
    print(f"\n=== {dataset_name.upper()} Experiment ===")
    print(f"n={n}, d={d}, num_classes={num_classes}")
    
    # Test different width regimes
    # gamma* for delta = d/n
    delta = d / n
    gamma_star = compute_gamma_star(delta)
    print(f"delta={delta:.2f}, gamma*={gamma_star:.2f}")
    
    results = []
    
    # Test supercritical (should have no spurious minima)
    gamma_supercritical = gamma_star * 2
    m_supercritical = int(gamma_supercritical * n)
    print(f"\n--- Supercritical: m={m_supercritical} (gamma={gamma_supercritical:.2f}) ---")
    
    global_loss, _ = find_global_minimum(train_loader, d, m_supercritical, num_classes, num_seeds=10)
    print(f"Global loss: {global_loss:.4f}")
    
    # Try to find spurious minima
    spurious_found = 0
    for seed in range(30):
        torch.manual_seed(seed + 100)
        
        model = TwoLayerReLU(d, m_supercritical, num_classes)
        with torch.no_grad():
            model.fc1.weight.normal_(0, 1 / np.sqrt(d))
            model.fc2.weight.normal_(0, 1 / np.sqrt(m_supercritical))
        
        model = train_model(model, train_loader, epochs=50)
        loss = evaluate_loss(model, train_loader)
        
        if loss > global_loss * 2:
            spurious_found += 1
    
    results.append({
        'regime': 'supercritical',
        'm': m_supercritical,
        'gamma': gamma_supercritical,
        'global_loss': global_loss,
        'spurious_found': spurious_found,
        'total_runs': 30
    })
    print(f"Spurious minima found: {spurious_found}/30")
    
    # Test subcritical (should have spurious minima theoretically)
    gamma_subcritical = gamma_star * 0.3
    m_subcritical = max(2, int(gamma_subcritical * n))
    print(f"\n--- Subcritical: m={m_subcritical} (gamma={gamma_subcritical:.2f}) ---")
    
    global_loss_sub, _ = find_global_minimum(train_loader, d, m_subcritical, num_classes, num_seeds=10)
    print(f"Global loss: {global_loss_sub:.4f}")
    
    spurious_found_sub = 0
    losses_sub = []
    for seed in range(30):
        torch.manual_seed(seed + 100)
        
        model = TwoLayerReLU(d, m_subcritical, num_classes)
        with torch.no_grad():
            model.fc1.weight.normal_(0, 1 / np.sqrt(d))
            model.fc2.weight.normal_(0, 1 / np.sqrt(m_subcritical))
        
        model = train_model(model, train_loader, epochs=50)
        loss = evaluate_loss(model, train_loader)
        losses_sub.append(loss)
        
        if loss > global_loss_sub * 2:
            spurious_found_sub += 1
    
    results.append({
        'regime': 'subcritical',
        'm': m_subcritical,
        'gamma': gamma_subcritical,
        'global_loss': global_loss_sub,
        'spurious_found': spurious_found_sub,
        'total_runs': 30,
        'loss_variance': float(np.var(losses_sub)),
        'loss_mean': float(np.mean(losses_sub)),
        'loss_std': float(np.std(losses_sub))
    })
    print(f"Spurious minima found: {spurious_found_sub}/30")
    print(f"Loss stats: mean={np.mean(losses_sub):.4f}, std={np.std(losses_sub):.4f}")
    
    # Test very narrow (undercapacity)
    m_narrow = 5
    print(f"\n--- Very narrow: m={m_narrow} (undercapacity) ---")
    
    global_loss_narrow, _ = find_global_minimum(train_loader, d, m_narrow, num_classes, num_seeds=10)
    print(f"Global loss: {global_loss_narrow:.4f}")
    
    spurious_found_narrow = 0
    losses_narrow = []
    for seed in range(30):
        torch.manual_seed(seed + 100)
        
        model = TwoLayerReLU(d, m_narrow, num_classes)
        with torch.no_grad():
            model.fc1.weight.normal_(0, 1 / np.sqrt(d))
            model.fc2.weight.normal_(0, 1 / np.sqrt(m_narrow))
        
        model = train_model(model, train_loader, epochs=50)
        loss = evaluate_loss(model, train_loader)
        losses_narrow.append(loss)
        
        if loss > global_loss_narrow * 1.5:
            spurious_found_narrow += 1
    
    results.append({
        'regime': 'undercapacity',
        'm': m_narrow,
        'gamma': m_narrow / n,
        'global_loss': global_loss_narrow,
        'spurious_found': spurious_found_narrow,
        'total_runs': 30,
        'loss_variance': float(np.var(losses_narrow)),
        'loss_mean': float(np.mean(losses_narrow)),
        'loss_std': float(np.std(losses_narrow))
    })
    print(f"Spurious minima found: {spurious_found_narrow}/30")
    print(f"Loss stats: mean={np.mean(losses_narrow):.4f}, std={np.std(losses_narrow):.4f}")
    
    # Save results
    os.makedirs("experiments/results", exist_ok=True)
    output_file = f"experiments/results/real_data_{dataset_name}.json"
    with open(output_file, "w") as f:
        json.dump({
            'dataset': dataset_name,
            'n': n,
            'delta': delta,
            'gamma_star': gamma_star,
            'results': results
        }, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar"])
    args = parser.parse_args()
    
    run_real_data_experiment(args.dataset)
