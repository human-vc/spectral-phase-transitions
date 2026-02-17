import json
import sys

try:
    with open("experiments/results/spectral_gap.json", "r") as f:
        data = json.load(f)
    
    if isinstance(data, list):
        print("wrapping list in dict...")
        new_data = {"results": data}
        with open("experiments/results/spectral_gap.json", "w") as f:
            json.dump(new_data, f)
        print("Fixed spectral_gap.json format.")
    else:
        print("spectral_gap.json is already a dict.")
        
except Exception as e:
    print(f"Error: {e}")
