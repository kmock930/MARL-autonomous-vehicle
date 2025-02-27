import random
import numpy as np
import joblib
import os

def generate_map(rowSize: int, colSize: int, num_obstacles: int, num_robots: int, tetherDist: int):
    # Initialize the map with free cells
    grid = np.zeros((rowSize, colSize), dtype=int)

    # Place obstacles randomly
    for _ in range(num_obstacles):
        while True:
            x, y = random.randint(0, rowSize-1), random.randint(0, colSize-1)
            if grid[x, y] == 0:
                grid[x, y] = 1
                break

    # Place robots randomly with roles
    robots = []
    roles = ['leader', 'follower']
    for i in range(num_robots):
        while True:
            x, y = random.randint(0, rowSize-1), random.randint(0, colSize-1)
            if grid[x, y] == 0:
                if i == 0 or all(max(abs(x - robot['position'][0]), abs(y - robot['position'][1])) <= tetherDist for robot in robots):
                    grid[x, y] = 2
                    robots.append({
                        'position': (x, y), 
                        'role': roles[i % len(roles)]
                    })
                    break

    # Place the target randomly
    while True:
        x, y = random.randint(0, rowSize-1), random.randint(0, colSize-1)
        if grid[x, y] == 0:
            grid[x, y] = 3
            target = (x, y)
            break

    return grid, robots, target

def print_map(grid):
    for row in grid:
        print(' '.join(str(cell) for cell in row))

def generate_sample_data(number: int):
    os.makedirs("sample_data", exist_ok=True)
    
    for currSampleId in range(number):
        size = random.randint(5, 10)
        num_obstacles = random.randint(5, 10)
        num_robots = random.randint(1, 3)
        grid, robots, target = generate_map(
            rowSize=size, 
            colSize=size, 
            num_obstacles=num_obstacles, 
            num_robots=num_robots,
            tetherDist=2
        )
        try:
            np.save(f"sample_data/sample_grid_{currSampleId+1}.npy", grid)
            joblib.dump(robots, f"sample_data/sample_robots_{currSampleId+1}.pkl")
            joblib.dump(target, f"sample_data/sample_target_{currSampleId+1}.pkl")
        except:
            print("Error in saving sample data")
            return False
    return True

if __name__ == "__main__":
    size = 10  # Size of the map
    num_obstacles = 15  # Number of obstacles
    num_robots = 2  # Number of robots
    tetherDist = 2  # Tether distance

    grid, robots, target = generate_map(
        rowSize=size, 
        colSize=size, 
        num_obstacles=num_obstacles, 
        num_robots=num_robots,
        tetherDist=tetherDist
    )
    print("Generated Map:")
    print_map(grid)
    print("\nRobots:", robots)
    print("Target:", target)

    generate_sample_data(10) # Generate 10 sample data