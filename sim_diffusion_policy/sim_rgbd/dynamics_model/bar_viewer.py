import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class ProbabilityVisualizer:
    def __init__(self, frames=100):
        # Initialization for the plot and animation parameters
        self.fig, self.ax = plt.subplots()
        self.frames = frames
        self.bars = self.ax.bar(['spillage'], [0])  # Single bar initialized at 0

        # Set the y-axis limit from 0 to 1
        self.ax.set_ylim(0, 1)
        self.ax.set_ylabel("Probability")

    def update_bar(self, frame):
        # Generate a new probability between 0 and 1 (replace with actual probability if needed)
        prob = np.random.rand()
        
        # Update the height of the bar
        self.bars[0].set_height(prob)
        
        # Optionally, update the title to display the current probability
        self.ax.set_title(f"Spillage Probability: {prob:.2f}")

    def animate(self):
        # Create animation
        ani = FuncAnimation(self.fig, self.update_bar, frames=self.frames, interval=500, repeat=False)
        plt.show()