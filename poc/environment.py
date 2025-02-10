import matplotlib.pyplot as plt

# Define grid dimensions
grid_size = (10, 10)

# Initialize the grid with empty cells
display_grid = [[" " for _ in range(grid_size[1])] for _ in range(grid_size[0])]

# Sample positions based on the provided image
r1_position = (1, 1)  # R1 position
r2_position = (8, 2)  # R2 position
target_position = (5, 7)  # Target position
obstacle_positions = [(2, 2), (3, 3), (5, 4), (7, 8), (4, 5), (9, 9), (0, 4), (5, 6), (4, 1), (6, 4), (3, 5), (2, 7)]

# Place elements on the grid
display_grid[r1_position[0]][r1_position[1]] = "R1"
display_grid[r2_position[0]][r2_position[1]] = "R2"
display_grid[target_position[0]][target_position[1]] = "T"

for obs in obstacle_positions:
    display_grid[obs[0]][obs[1]] = "*"

# Plotting the environment
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(0, grid_size[1])
ax.set_ylim(0, grid_size[0])
ax.set_aspect('equal')

# Draw gridlines with borders around all cells
for x in range(grid_size[1] + 1):
    ax.axvline(x, color='black', lw=1)  # Thicker lines for better visibility
for y in range(grid_size[0] + 1):
    ax.axhline(y, color='black', lw=1)

# Draw the grid elements
for i in range(grid_size[0]):
    for j in range(grid_size[1]):
        ax.text(j + 0.5, grid_size[0] - i - 0.5, display_grid[i][j], va='center', ha='center', fontsize=12)

# Adding column and row labels
columns = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
rows = [str(i + 1) for i in range(grid_size[0])]

# Draw column labels at the topmost part of the grid
for idx, label in enumerate(columns):
    ax.text(idx + 0.5, -0.7, label, va='center', ha='center', fontsize=12, fontweight='bold')

# Draw row labels
for idx, label in enumerate(rows):
    ax.text(-0.7, idx + 0.5, label, va='center', ha='center', fontsize=12, fontweight='bold')

# Configure plot appearance
plt.gca().invert_yaxis()
plt.axis('off')

# Save the plot to a file
plt.savefig('output.png', bbox_inches='tight')
plt.show()