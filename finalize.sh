#!/bin/bash
echo "Copying results to Downloads..."
cp experiments/results/phase_boundary_empirical.pdf ~/Downloads/
cp experiments/results/spectral_density.pdf ~/Downloads/

echo "Fixing spectral_gap.json format..."
python3 experiments/fix_json.py

echo "Generating additional figures..."
python3 experiments/plot_results.py --phase-boundary dummy.json --spectral-gap experiments/results/spectral_gap.json --output-dir experiments/results/

echo "Committing to git..."
git add experiments/results/
git commit -m "Full experiment results"
git push origin main
echo "Done!"
