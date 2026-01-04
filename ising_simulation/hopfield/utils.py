
import numpy as np
from typing import Dict

def create_letter_pattern(letter: str, size: int = 10) -> np.ndarray:
    """
    Creates a binary pattern (+1/-1) for a specific letter.
    Currently supports 'Y', 'E', 'H', 'I', 'A' on a 10x10 grid.
    
    Args:
        letter (str): The character to generate.
        size (int): Grid size (width=height). Defaults to 10.
        
    Returns:
        np.ndarray: Flattened 1D array of size (size*size) with values {+1, -1}.
    """
    letter = letter.upper()
    
    # Initialize background as -1
    grid = np.full((size, size), -1, dtype=int)
    
    # Helper to clean up coordinate definitions
    # Coordinates are (row, col)
    # 10x10 grid -> indices 0..9
    
    if letter == 'Y':
        # Y shape
        # Top-left arm
        for i in range(5):
            grid[i, i] = 1
        # Top-right arm
        for i in range(5):
            grid[i, size - 1 - i] = 1
        # Stem
        for i in range(5, size):
            grid[i, size // 2] = 1
            if size % 2 == 0: # make stem thicker for even sizes
                grid[i, size // 2 - 1] = 1
                
    elif letter == 'E':
        # Left bar
        grid[:, 1:3] = 1
        # Top bar
        grid[0:2, 1:] = 1
        # Middle bar
        grid[size//2 - 1 : size//2 + 1, 1:] = 1
        # Bottom bar
        grid[-2:, 1:] = 1
        
    elif letter == 'H':
        # Left bar
        grid[:, 1:3] = 1
        # Right bar
        grid[:, -3:-1] = 1
        # Middle bar
        grid[size//2 - 1 : size//2 + 1, :] = 1
        
    elif letter == 'I':
        # Top bar
        grid[0:2, :] = 1
        # Bottom bar
        grid[-2:, :] = 1
        # Stem
        center = size // 2
        grid[:, center-1:center+1] = 1
        
    elif letter == 'A':
        # Left diagonalish - simplified to straight lines for low res
        # Left Side
        grid[1:, 1:3] = 1
        # Right Side
        grid[1:, -3:-1] = 1
        # Top
        grid[0:2, 1:-1] = 1
        # Middle
        grid[size//2 - 1 : size//2 + 1, 1:-1] = 1
        
    else:
        raise ValueError(f"Letter '{letter}' not implemented. Supported: Y, E, H, I, A")
        
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
