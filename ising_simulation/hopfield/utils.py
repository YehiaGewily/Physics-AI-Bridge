
import numpy as np
from typing import Dict

def create_letter_pattern(letter: str, size: int = 10) -> np.ndarray:
    """
    Creates a binary pattern (+1/-1) for a specific letter.
    Currently supports 'N', 'O', 'V', 'E', 'R' on a 10x10 grid.
    
    Args:
        letter (str): The character to generate.
        size (int): Grid size (width=height). Must be 10. Defaults to 10.
        
    Returns:
        np.ndarray: Flattened 1D array of shape (100,) with dtype=np.int8
                     and values in {-1, +1}.
    """
    letter = letter.upper()
    
    if size != 10:
        raise ValueError(f"Letter patterns are designed for size=10 only. Got size={size}.")
    
    patterns: Dict[str, list] = {
        'N': [
            [+1,+1,-1,-1,-1,-1,-1,-1,+1,+1],
            [+1,+1,+1,-1,-1,-1,-1,-1,+1,+1],
            [+1,+1,+1,+1,-1,-1,-1,-1,+1,+1],
            [+1,+1,-1,+1,+1,-1,-1,-1,+1,+1],
            [+1,+1,-1,-1,+1,+1,-1,-1,+1,+1],
            [+1,+1,-1,-1,-1,+1,+1,-1,+1,+1],
            [+1,+1,-1,-1,-1,-1,+1,+1,+1,+1],
            [+1,+1,-1,-1,-1,-1,-1,+1,+1,+1],
            [+1,+1,-1,-1,-1,-1,-1,-1,+1,+1],
            [+1,+1,-1,-1,-1,-1,-1,-1,+1,+1],
        ],
        'O': [
            [-1,-1,+1,+1,+1,+1,+1,+1,-1,-1],
            [-1,+1,+1,+1,+1,+1,+1,+1,+1,-1],
            [+1,+1,-1,-1,-1,-1,-1,-1,+1,+1],
            [+1,+1,-1,-1,-1,-1,-1,-1,+1,+1],
            [+1,+1,-1,-1,-1,-1,-1,-1,+1,+1],
            [+1,+1,-1,-1,-1,-1,-1,-1,+1,+1],
            [+1,+1,-1,-1,-1,-1,-1,-1,+1,+1],
            [+1,+1,-1,-1,-1,-1,-1,-1,+1,+1],
            [-1,+1,+1,+1,+1,+1,+1,+1,+1,-1],
            [-1,-1,+1,+1,+1,+1,+1,+1,-1,-1],
        ],
        'V': [
            [+1,+1,-1,-1,-1,-1,-1,-1,+1,+1],
            [+1,+1,-1,-1,-1,-1,-1,-1,+1,+1],
            [+1,+1,-1,-1,-1,-1,-1,-1,+1,+1],
            [-1,+1,+1,-1,-1,-1,-1,+1,+1,-1],
            [-1,+1,+1,-1,-1,-1,-1,+1,+1,-1],
            [-1,-1,+1,+1,-1,-1,+1,+1,-1,-1],
            [-1,-1,+1,+1,-1,-1,+1,+1,-1,-1],
            [-1,-1,-1,+1,+1,+1,+1,-1,-1,-1],
            [-1,-1,-1,+1,+1,+1,+1,-1,-1,-1],
            [-1,-1,-1,-1,+1,+1,-1,-1,-1,-1],
        ],
        'E': [
            [+1,+1,+1,+1,+1,+1,+1,+1,-1,-1],
            [+1,+1,-1,-1,-1,-1,-1,-1,-1,-1],
            [+1,+1,-1,-1,-1,-1,-1,-1,-1,-1],
            [+1,+1,-1,-1,-1,-1,-1,-1,-1,-1],
            [+1,+1,+1,+1,+1,+1,-1,-1,-1,-1],
            [+1,+1,+1,+1,+1,+1,-1,-1,-1,-1],
            [+1,+1,-1,-1,-1,-1,-1,-1,-1,-1],
            [+1,+1,-1,-1,-1,-1,-1,-1,-1,-1],
            [+1,+1,-1,-1,-1,-1,-1,-1,-1,-1],
            [+1,+1,+1,+1,+1,+1,+1,+1,-1,-1],
        ],
        'R': [
            [+1,+1,+1,+1,+1,+1,+1,-1,-1,-1],
            [+1,+1,-1,-1,-1,-1,-1,+1,+1,-1],
            [+1,+1,-1,-1,-1,-1,-1,-1,+1,+1],
            [+1,+1,-1,-1,-1,-1,-1,-1,+1,+1],
            [+1,+1,-1,-1,-1,-1,-1,+1,+1,-1],
            [+1,+1,+1,+1,+1,+1,+1,-1,-1,-1],
            [+1,+1,-1,-1,+1,+1,-1,-1,-1,-1],
            [+1,+1,-1,-1,-1,+1,+1,-1,-1,-1],
            [+1,+1,-1,-1,-1,-1,+1,+1,-1,-1],
            [+1,+1,-1,-1,-1,-1,-1,+1,+1,-1],
        ],
    }
    
    if letter not in patterns:
        raise ValueError(
            f"Letter '{letter}' not implemented. "
            f"Supported: {', '.join(sorted(patterns.keys()))}"
        )
    
    grid = np.array(patterns[letter], dtype=np.int8)
    return grid.flatten()

def corrupt_pattern(pattern: np.ndarray, corruption_rate: float) -> np.ndarray:
    """
    Randomly flips a fraction of spins in the pattern.
    
    Args:
        pattern (np.ndarray): The original 1D binary pattern.
        corruption_rate (float): Fraction of spins to flip (0.0 to 1.0).
        
    Returns:
        np.ndarray: The corrupted pattern.
    """
    if not 0 <= corruption_rate <= 1:
        raise ValueError("Corruption rate must be between 0 and 1.")
        
    corrupted = pattern.copy()
    n_flips = int(len(pattern) * corruption_rate)
    
    # Choose random indices to flip
    flip_indices = np.random.choice(len(pattern), size=n_flips, replace=False)
    
    # Flip values: x * -1
    corrupted[flip_indices] *= -1
    
    return corrupted

def pattern_to_image(pattern: np.ndarray, size: int) -> np.ndarray:
    """
    Reshapes a flattened pattern into a 2D image grid.
    
    Args:
        pattern (np.ndarray): Flattened 1D array.
        size (int): The linear dimension of the grid (L).
        
    Returns:
        np.ndarray: 2D array of shape (size, size).
    """
    if len(pattern) != size * size:
        raise ValueError(f"Pattern length {len(pattern)} does not match size {size}x{size}={size*size}")
        
    return pattern.reshape((size, size))
