import sys
import os
import pygame
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from collections import deque

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.ising_model import IsingSimulation

class Slider:
    def __init__(self, x, y, w, h, min_val, max_val, initial_val, label):
        self.rect = pygame.Rect(x, y, w, h)
        self.min_val = min_val
        self.max_val = max_val
        self.val = initial_val
        self.label = label
        self.dragging = False
        self.knob_rect = pygame.Rect(x, y - 5, 10, h + 10)
        self.update_knob_pos()

    def update_knob_pos(self):
        ratio = (self.val - self.min_val) / (self.max_val - self.min_val)
        self.knob_rect.centerx = self.rect.left + ratio * self.rect.width

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.knob_rect.collidepoint(event.pos) or self.rect.collidepoint(event.pos):
                self.dragging = True
                self.update_val(event.pos[0])
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION:
            if self.dragging:
                self.update_val(event.pos[0])

    def update_val(self, mouse_x):
        mouse_x = max(self.rect.left, min(mouse_x, self.rect.right))
        ratio = (mouse_x - self.rect.left) / self.rect.width
        self.val = self.min_val + ratio * (self.max_val - self.min_val)
        self.update_knob_pos()

    def draw(self, screen, font):
        pygame.draw.rect(screen, (200, 200, 200), self.rect)
        pygame.draw.rect(screen, (100, 100, 255), self.knob_rect)
        label_surf = font.render(f"{self.label}: {self.val:.2f}", True, (255, 255, 255))
        screen.blit(label_surf, (self.rect.left, self.rect.top - 25))

def main():
    pygame.init()
    
    # Window Layout
    WINDOW_WIDTH = 1200
    WINDOW_HEIGHT = 800
    
    # Grid Settings
    GRID_SIZE = 256
    GRID_DISPLAY_SIZE = 500
    
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Ising Model: Thermodynamics Dashboard")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('Arial', 16)
    
    # Initialize Physics
    model = IsingSimulation(size=GRID_SIZE, temperature=2.27, B=0.0)
    
    # UI Controls
    # Sliders on the right side top
    temp_slider = Slider(600, 50, 300, 10, 0.1, 5.0, 2.27, "Temperature (T)")
    field_slider = Slider(600, 120, 300, 10, -2.0, 2.0, 0.0, "Magnetic Field (B)")
    
    # Matplotlib Setup for 4-panel plot
    matplotlib.use("Agg")
    fig, axes = plt.subplots(2, 2, figsize=(6, 5), dpi=100)
    fig.patch.set_facecolor('#222222') # Dark background
    
    # Flatten axes for easy access: 0=E, 1=M, 2=Cv, 3=Chi
    ax_list = axes.flatten()
    titles = ["Energy", "Magnetization", "Specific Heat (Cv)", "Susceptibility (Chi)"]
    colors = ['cyan', 'magenta', 'orange', 'lime']
    
    # Histories
    hist_len = 200
    data_history = {
        "E": deque(maxlen=hist_len),
        "M": deque(maxlen=hist_len),
        "Cv": deque(maxlen=hist_len),
        "Chi": deque(maxlen=hist_len)
    }

    running = True
    frame_count = 0
    
    while running:
        # Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            temp_slider.handle_event(event)
            field_slider.handle_event(event)
            
        # Update Model Parameters
        model.set_temperature(temp_slider.val)
        model.B = field_slider.val
        
        # Physics Steps
        model.metropolis_step(steps_per_sweep=1)
        
        # Collect Stats
        stats = model.get_statistics()
        # Normalize E and M for display
        N = GRID_SIZE * GRID_SIZE
        data_history["E"].append(stats["mean_E"] / N)
        data_history["M"].append(stats["mean_M"] / N)
        data_history["Cv"].append(stats["Cv"] / N) # Intensive property
        data_history["Chi"].append(stats["Chi"] / N) # Intensive property
        
        # Rendering
        screen.fill((30, 30, 30))
        
        # 1. Draw Grid (Left Side)
        grid_arr = model.grid
        rgb_grid = np.zeros((GRID_SIZE, GRID_SIZE, 3), dtype=np.uint8)
        mask_up = (grid_arr == 1)
        rgb_grid[mask_up] = [255, 215, 0]   # Gold
        rgb_grid[~mask_up] = [50, 50, 150] # Blue
        
        # Transpose and Scale
        surf_data = np.transpose(rgb_grid, (1, 0, 2))
        grid_surf = pygame.surfarray.make_surface(surf_data)
        grid_surf = pygame.transform.scale(grid_surf, (GRID_DISPLAY_SIZE, GRID_DISPLAY_SIZE))
        screen.blit(grid_surf, (20, 20))
        
        # Draw Legend
        legend_y = 540
        # Gold Box
        pygame.draw.rect(screen, (255, 215, 0), (20, legend_y, 20, 20))
        screen.blit(font.render("Spin UP (+1)", True, (200, 200, 200)), (45, legend_y))
        
        # Blue Box
        pygame.draw.rect(screen, (50, 50, 150), (140, legend_y, 20, 20))
        screen.blit(font.render("Spin DOWN (-1)", True, (200, 200, 200)), (165, legend_y))
        
        # 2. Draw Controls
        temp_slider.draw(screen, font)
        field_slider.draw(screen, font)
        
        # 3. Draw Info Text
        info_x = 600
        info_y = 160
        lines = [
            f"Net Magnetization: {stats['mean_M']/N:.3f}",
            f"Mean Energy: {stats['mean_E']/N:.3f}",
            f"Specific Heat (Cv): {stats['Cv']/N:.3f}",
            f"Susceptibility (Chi): {stats['Chi']/N:.3f}"
        ]
        for i, line in enumerate(lines):
            t_s = font.render(line, True, (200, 200, 200))
            screen.blit(t_s, (info_x, info_y + i*20))

        # 4. Update plots (every 10 frames)
        if frame_count % 10 == 0 and len(data_history["E"]) > 2:
            for i, (key, ax) in enumerate(zip(["E", "M", "Cv", "Chi"], ax_list)):
                ax.clear()
                ax.plot(data_history[key], color=colors[i], linewidth=1.5)
                ax.set_title(titles[i], color='white', fontsize=8)
                ax.set_facecolor('#333333')
                ax.tick_params(colors='white', labelsize=6)
                ax.grid(True, linestyle='--', alpha=0.2)
            
            plt.tight_layout()
            canvas = FigureCanvas(fig)
            canvas.draw()
            raw_data = canvas.buffer_rgba()
            size = canvas.get_width_height()
            plot_surf = pygame.image.frombuffer(raw_data, size, "RGBA")
            last_plot_surf = plot_surf
        
        if 'last_plot_surf' in locals():
            screen.blit(last_plot_surf, (550, 250))
            
        pygame.display.flip()
        clock.tick(60)
        frame_count += 1

    pygame.quit()
    plt.close()

if __name__ == "__main__":
    main()
