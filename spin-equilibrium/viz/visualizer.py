import sys
import os
import pygame
import numpy as np

# Add project root (spin-equilibrium) to path to import core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.ising_model import IsingSimulation

# Constants
WIDTH, HEIGHT = 800, 800
GRID_SIZE = 100 # 100x100 grid
CELL_SIZE = WIDTH // GRID_SIZE
FPS = 60

# Colors
COLOR_UP = (255, 255, 255) # White for spin +1
COLOR_DOWN = (0, 0, 0)     # Black for spin -1
TEXT_COLOR = (255, 0, 0)   # Red for text

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Ising Model Simulation - Phase 1")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('Arial', 24)

    # Initialize Physics Model
    # Critical Temperature ~ 2.269
    model = IsingSimulation(size=GRID_SIZE, temperature=2.269)
    
    running = True
    paused = False

    while running:
        # 1. Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    model.temperature += 0.1
                elif event.key == pygame.K_DOWN:
                    model.temperature = max(0.1, model.temperature - 0.1)
                elif event.key == pygame.K_r:
                    # Reset grid
                    model.grid = np.random.choice([-1, 1], size=(GRID_SIZE, GRID_SIZE))
                elif event.key == pygame.K_SPACE:
                    paused = not paused

        # 2. Physics Update
        if not paused:
            # multiple steps per frame to speed up equilibration
            model.metropolis_step(steps_per_sweep=GRID_SIZE * GRID_SIZE)

        # 3. Rendering
        screen.fill((100, 100, 100)) # Grey background
        
        # Lock surface for faster pixel access could be optimized, 
        # but rect drawing is fine for 100x100
        
        # Create a surface from the grid for faster blitting?
        # Or just draw rects. 100x100 = 10,000 rects per frame. Might be slow in pure python.
        # fast drawing: use pygame.surfarray
        
        # Map spins to colors: -1 -> 0 (Black), 1 -> 255 (White)
        # (grid + 1) / 2 * 255
        
        img_array = ((model.grid + 1) // 2 * 255).astype(np.uint8)
        # Valid assumption: grid is (rows, cols). surface expects (width, height) -> (cols, rows)
        # So we transpose
        img_array = img_array.T
        
        # Make RGB (3 channels)
        img_rgb = np.dstack((img_array, img_array, img_array))
        
        surf = pygame.surfarray.make_surface(img_rgb)
        surf = pygame.transform.scale(surf, (WIDTH, HEIGHT))
        screen.blit(surf, (0, 0))

        # 4. HUD
        temp_text = font.render(f"Temp: {model.temperature:.2f} (Tc ~ 2.27)", True, TEXT_COLOR)
        controls_text = font.render("UP/DOWN: Temp | R: Reset | SPACE: Pause", True, TEXT_COLOR)
        screen.blit(temp_text, (10, 10))
        screen.blit(controls_text, (10, 40))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()
