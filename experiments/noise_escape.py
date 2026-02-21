"""
Experiment 4: Test effect of temperature/noise on escaping spurious basins.

Adds noise to initialization and measures how it affects
escape from spurious minima vs convergence to global.
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


def train_with_noise(model, X, y, noise_scale=0, lr=0.01, epochs=200):
    """Train with optional noise injection during initialization."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X)
        loss = nn.functional.mse_loss(pred, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    return losses


def find_minima(X, y, d, m, num_seeds=50):
    """Find all distinct minima by trying many random initializations."""
    all_losses = []
    all_states = []
    
    for seed in range(num_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model = TwoLayerReLU(d, m)
        with torch.no_grad():
            model.fc1.weight.normal_(0, 1 / np.sqrt(d))
            model.fc2.weight.normal_(0, 1 / np.sqrt(m))
        
        losses = train_with_noise(model, X, y, epochs=300)
        final_loss = losses[-1]
        
        all_losses.append(final_loss)
        all_states.append({k: v.clone() for k, v in model.state_dict().items()})
    
    # Cluster losses to identify distinct basins
    losses_array = np.array(all_losses)
    
    # Find unique loss values (within tolerance)
    unique_basins = []
    tolerance = 0.01
    
    for i, loss in enumerate(losses_array):
        is_new = True
        for ub_loss, _ in unique_basins:
            if abs(loss - ub_loss) < tolerance:
                is_new = False
                break
        
        if is_new:
            unique_basins.append((loss, all_states[i]))
    
    unique_basins.sort(key=lambda x: x[0])
    
    return unique_basins, losses_array


def measure_noise_escape(X, y, d, m, spurious_state, global_state, noise_levels):
    """
    Measure how noise affects escape from spurious basins.
    """
    results = []
    
    global_loss = train_with_noise(
        TwoLayerReLU(d, m), X, y, epochs=100
    )[-1]
    
    # Slightly perturb global to get reference
    global_model = TwoLayerReLU(d, m)
    global_model.load_state_dict({k: v.clone() for k, v in global_state.items()})
    global_loss = train_with_noise(global_model, X, y, epochs=100)[-1]
    
    print(f"Global loss: {global_loss:.6f}")
    
    # Test each noise level
    for noise_scale in noise_levels:
        escaped = 0
        trapped = 0
        final_losses = []
        
        for seed in range(30):
            torch.manual_seed(seed * 100 + int(noise_scale * 1000))
            
            model = TwoLayerReLU(d, m)
            
            # Initialize near spurious with noise
            with torch.no_grad():
                # Start from spurious minimum
                model.fc1.weight.copy_(spurious_state['fc1.weight'])
                model.fc2.weight.copy_(spurious_state['fc2.weight'])
                
                # Add noise
                noise = torch.randn_like(model.fc1.weight) * noise_scale
                model.fc1.weight.add_(noise)
                
                # Re-randomize output layer to increase chance of escaping
                model.fc2.weight.normal_(0, 1 / np.sqrt(m))
            
            # Train
            losses = train_with_noise(model, X, y, epochs=200)
            final_loss = losses[-1]
            final_losses.append(final_loss)
            
            # Check if escaped to global basin
            if final_loss < global_loss * 2:
                escaped += 1
            else:
                trapped += 1
        
        escape_rate = escaped / 30
        mean_loss = np.mean(final_losses)
        
        results.append({
            'noise_scale': noise_scale,
            'escaped': escaped,
            'trapped': trapped,
            'escape_rate': escape_rate,
            'mean_final_loss': float(mean_loss)
        })
        
        print(f"noise={noise_scale:.3f}: escaped={escaped}/30 ({escape_rate:.1%}), mean_loss={mean_loss:.4f}")
    
    return results


def run_temperature_experiment():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    
    # Setup: 2D toy case with known spurious minima
    n = 80
    d = 40
    m = 5  # Narrow - can't fully interpolate
    
    torch.manual_seed(42)
    np.random.seed(42)
    X = torch.randn(n, d) * 0.5
    y = torch.randn(n, 1) * 2  # Random labels
    
    print(f"Setup: n={n}, d={d}, m={m}")
    
    # Find distinct basins
    print("\nFinding distinct minima...")
    basins, all_losses = find_minima(X, y, d, m, num_seeds=100)
    
    print(f"Found {len(basins)} distinct basins:")
    for i, (loss, _) in enumerate(basins):
        print(f"  Basin {i}: loss={loss:.6f}")
    
    if len(basins) < 2:
        print("Only one basin found - trying different parameters...")
        # Try with even narrower network
        m = 3
        print(f"Retrying with m={m}...")
        
        basins, all_losses = find_minima(X, y, d, m, num_seeds=100)
        print(f"Found {len(basins)} distinct basins")
    
    if len(basins) >= 2:
        # Identify global (lowest) and spurious (higher) basins
        global_state = basins[0][1]
        global_loss = basins[0][0]
        
        # Use highest loss basin as spurious
        spurious_state = basins[-1][1]
        spurious_loss = basins[-1][0]
        
        print(f"\nGlobal basin: loss={global_loss:.6f}")
        print(f"Spurious basin: loss={spurious_loss:.6f}")
        
        # Test noise levels
        noise_levels = [0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
        
        print("\nTesting noise escape...")
        results = measure_noise_escape(
            X, y, d, m, spurious_state, global_state, noise_levels
        )
        
        # Save results
        os.makedirs("experiments/results", exist_ok=True)
        with open("experiments/results/noise_escape.json", "w") as f:
            json.dump({
                'n': n,
                'd': d,
                'm': m,
                'global_loss': global_loss,
                'spurious_loss': spurious_loss,
                'num_basins': len(basins),
                'noise_experiment': results
            }, f, indent=2)
        
        print("\nResults saved to experiments/results/noise_escape.json")
    else:
        print("Could not find multiple basins with these parameters")


if __name__ == "__main__":
    run_temperature_experiment()
