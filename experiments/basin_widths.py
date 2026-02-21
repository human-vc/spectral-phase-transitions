"""
Experiment 1: Quantify basin widths for spurious minima.

Measures how close initialization needs to be to a spurious minimum
to get trapped vs escape to global minimum. Connects to the
"exponentially small basin measure" theory.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import os


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
        return self.fc2(F.relu(self.fc1(x)))


def train_with_early_stop(model, X, y, lr=0.01, epochs=2000, target_loss=None):
    """Train until convergence or early stop."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X)
        loss = nn.functional.mse_loss(pred, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if target_loss is not None and loss.item() <= target_loss:
            break
    
    return losses


def find_global_minimum(X, y, d, m, num_seeds=20):
    """Find the global minimum loss across multiple random inits."""
    best_loss = float('inf')
    best_model_state = None
    
    for seed in range(num_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model = TwoLayerReLU(d, m)
        with torch.no_grad():
            model.fc1.weight.normal_(0, 1 / np.sqrt(d))
            model.fc2.weight.normal_(0, 1 / np.sqrt(m))
        
        losses = train_with_early_stop(model, X, y, lr=0.01, epochs=1000)
        final_loss = losses[-1]
        
        if final_loss < best_loss:
            best_loss = final_loss
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    return best_loss, best_model_state


def measure_basin_width(X, y, d, m, spurious_center, global_min_state, num_init=50):
    """
    Measure basin width by varying distance from spurious center.
    Returns fraction of runs that get trapped in spurious vs escape to global.
    """
    results = []
    global_loss, _ = find_global_minimum(X, y, d, m, num_seeds=5)
    
    # Vary initialization distance from spurious center
    distances = np.logspace(-3, 1, num_init)  # 0.001 to 10
    
    for dist in distances:
        trapped_count = 0
        escape_count = 0
        
        for seed in range(10):
            torch.manual_seed(seed * 1000 + int(dist * 100))
            
            model = TwoLayerReLU(d, m)
            
            # Initialize near spurious center with perturbation
            with torch.no_grad():
                # Direction from spurious to global
                spurious_flat = spurious_center['fc1.weight']
                global_flat = global_min_state['fc1.weight']
                direction = global_flat - spurious_flat
                direction = direction / (direction.norm() + 1e-8)
                
                # Init at distance dist from spurious toward global
                init_weight = spurious_flat + direction * dist
                model.fc1.weight.copy_(init_weight)
                model.fc2.weight.normal_(0, 1 / np.sqrt(m))
            
            # Train and check final loss
            losses = train_with_early_stop(model, X, y, lr=0.01, epochs=500)
            final_loss = losses[-1]
            
            # If close to global loss, escaped. If much higher, trapped.
            if final_loss < global_loss * 1.5:
                escape_count += 1
            else:
                trapped_count += 1
        
        results.append({
            'distance': float(dist),
            'trapped': trapped_count,
            'escaped': escape_count,
            'trap_rate': trapped_count / 10
        })
        print(f"dist={dist:.3f}: trapped={trapped_count}, escaped={escape_count}")
    
    return results


def run_basin_experiment():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    
    # Setup: 2D toy case where we know spurious minima exist
    # m << d to create undercapacity
    n = 50
    d = 20
    m = 2  # Very narrow - can't interpolate
    
    # Random data
    torch.manual_seed(42)
    np.random.seed(42)
    X = torch.randn(n, d) * 0.5
    
    # Random labels (non-realizable - guarantees spurious minima)
    y = torch.randn(n, 1) * 2
    
    print(f"Setup: n={n}, d={d}, m={m}")
    
    # Find global minimum
    global_loss, global_state = find_global_minimum(X, y, d, m, num_seeds=20)
    print(f"Global min loss: {global_loss:.6f}")
    
    # Find a spurious minimum by trying many inits
    spurious_losses = []
    spurious_states = []
    
    for seed in range(100):
        torch.manual_seed(seed)
        
        model = TwoLayerReLU(d, m)
        with torch.no_grad():
            model.fc1.weight.normal_(0, 1 / np.sqrt(d))
            model.fc2.weight.normal_(0, 1 / np.sqrt(m))
        
        losses = train_with_early_stop(model, X, y, lr=0.01, epochs=500)
        final_loss = losses[-1]
        
        # If not reaching global, might be spurious
        if final_loss > global_loss * 2:
            spurious_losses.append(final_loss)
            spurious_states.append({k: v.clone() for k, v in model.state_dict().items()})
    
    if not spurious_states:
        print("No spurious minima found - trying harder...")
        # Try even more random inits
        for seed in range(100, 200):
            torch.manual_seed(seed)
            
            model = TwoLayerReLU(d, m)
            # Try different init scales
            with torch.no_grad():
                model.fc1.weight.uniform_(-3, 3)
                model.fc2.weight.normal_(0, 0.5)
            
            losses = train_with_early_stop(model, X, y, lr=0.001, epochs=1000)
            final_loss = losses[-1]
            
            if final_loss > global_loss * 2:
                spurious_losses.append(final_loss)
                spurious_states.append({k: v.clone() for k, v in model.state_dict().items()})
    
    print(f"Found {len(spurious_states)} potential spurious minima")
    print(f"Spurious losses: min={min(spurious_losses):.4f}, max={max(spurious_losses):.4f}")
    
    if spurious_states:
        # Use the worst spurious minimum
        worst_idx = np.argmax(spurious_losses)
        spurious_center = spurious_states[worst_idx]
        
        print(f"\nMeasuring basin width from spurious (loss={spurious_losses[worst_idx]:.4f}) to global...")
        
        results = measure_basin_width(X, y, d, m, spurious_center, global_state)
        
        os.makedirs("experiments/results", exist_ok=True)
        with open("experiments/results/basin_widths.json", "w") as f:
            json.dump({
                'global_loss': global_loss,
                'spurious_loss': spurious_losses[worst_idx],
                'basin_measurements': results
            }, f, indent=2)
        
        print("\nBasin width experiment complete!")
        print(f"Results saved to experiments/results/basin_widths.json")
    else:
        print("Could not find spurious minimum - try different parameters")


if __name__ == "__main__":
    run_basin_experiment()
