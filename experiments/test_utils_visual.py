
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ising_simulation.hopfield.utils import create_letter_pattern, corrupt_pattern, pattern_to_image

def test_utils_visual():
    letters = ['Y', 'E', 'H', 'I', 'A']
    size = 10
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    for i, letter in enumerate(letters):
        # 1. Create
        pat = create_letter_pattern(letter, size)
        
        # 2. Corrupt (30% noise)
        noisy_pat = corrupt_pattern(pat, corruption_rate=0.3)
        
        # 3. Visualize Original
        ax_orig = axes[0, i]
        ax_orig.imshow(pattern_to_image(pat, size), cmap='gray', vmin=-1, vmax=1)
        ax_orig.set_title(f"Original: {letter}")
        ax_orig.axis('off')
        
        # 4. Visualize Corrupted
        ax_noise = axes[1, i]
        ax_noise.imshow(pattern_to_image(noisy_pat, size), cmap='gray', vmin=-1, vmax=1)
        ax_noise.set_title(f"Corrupted (30%)")
        ax_noise.axis('off')
        
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), '../results/figures/test_utils_visual.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Visualization saved to: {output_path}")

if __name__ == "__main__":
    test_utils_visual()
