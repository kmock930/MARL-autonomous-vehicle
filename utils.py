from constants import OBJECT_ENCODING
import numpy as np

# Modify the `is_invalid` function to check roles using the Agent class
def is_invalid(new_pos: tuple[int, int], agents: list, grid_map: np.ndarray) -> int | None:
    if not isinstance(grid_map, np.ndarray) or grid_map.ndim != 2:
        raise ValueError("grid_map must be a 2D numpy array")

    rowSize, colSize = grid_map.shape

    # Check if the new position is within the grid
    if not (0 <= new_pos[0] < rowSize and 0 <= new_pos[1] < colSize):
        print("Out of Bounds")  # Debugging message
        return OBJECT_ENCODING.OUT_OF_BOUNDS

    # Check if the new position is occupied by another agent
    for agent in agents:
        if agent["position"] == new_pos:
            # Collision with agent
            return OBJECT_ENCODING.AGENT

    # Check if the new position is a hard obstacle
    if grid_map[new_pos[0], new_pos[1]] in [OBJECT_ENCODING.OBSTACLE_HARD]:
        print("Obstacle Collision")  # Debugging message
        return OBJECT_ENCODING.OBSTACLE_HARD

    print("Valid Move")  # Debugging message
    return None