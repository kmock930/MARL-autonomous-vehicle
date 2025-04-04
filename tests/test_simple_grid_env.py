import unittest
import sys
import os
import numpy as np
SIMPLEGRID_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'gym-simplegrid', 'gym_simplegrid', 'envs'))
sys.path.append(SIMPLEGRID_PATH)
from simple_grid import SimpleGridEnv
ROOT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..")
sys.path.append(ROOT_PATH)
from constants import ACTION_SPACE

class TestSimpleGridEnv(unittest.TestCase):
    def tearDown(self):
        if hasattr(self, 'env'):
            self.env.close()

    def test_samplevalidStatexy(self):
        # Define a simple map
        obstacle_map = ["0000", "0101", "0001", "1000"]
        agent_map = ["0030", "0000", "0000", "0000"]
        target_map = ["0000", "0000", "0000", "0004"]

        # Initialize the environment
        self.env = SimpleGridEnv(
            obstacle_map=obstacle_map, 
            agent_map=agent_map, 
            target_map=target_map, 
            render_mode="human",
            rowSize=4,
            colSize=4,
            num_soft_obstacles=2,
            num_hard_obstacles=2,
            num_robots=1,
            tetherDist=2,
            num_leaders=1,
            num_target=1
        )
        
        valid_state:tuple = self.env.sample_valid_state_xy()
        self.assertEqual(type(valid_state), tuple)
        self.assertEqual(len(valid_state), 2)
    
    def test_parseObstacleMap_listFormat(self):
        # Define a simple map
        obstacle_map = ["0000", "0101", "0001", "1000"]
        agent_map = ["0030", "0000", "0000", "0000"]
        target_map = ["0000", "0000", "0000", "0004"]

        # Initialize the environment
        self.env = SimpleGridEnv(
            obstacle_map=obstacle_map, 
            agent_map=agent_map, 
            target_map=target_map, 
            render_mode="human",
            rowSize=4,
            colSize=4,
            num_soft_obstacles=2,
            num_hard_obstacles=2,
            num_robots=1,
            tetherDist=2,
            num_leaders=1,
            num_target=1
        )
        
        parsed_obstacle_map = self.env.parse_obstacle_map(
            obstacle_map=obstacle_map
        )
        self.assertEqual(type(parsed_obstacle_map), np.ndarray)
        self.assertEqual(len(parsed_obstacle_map), 4)

    def test_parseObstacleMap_strFormat(self):
        # Define a simple map
        obstacle_map = "0000\n0101\n0001\n1000"
        agent_map = "0030\n0000\n0000\n0000"
        target_map = "0000\n0000\n0000\n0004"

        # Initialize the environment
        self.env = SimpleGridEnv(
            obstacle_map=obstacle_map, 
            agent_map=agent_map, 
            target_map=target_map, 
            render_mode="human",
            rowSize=4,
            colSize=4,
            num_soft_obstacles=2,
            num_hard_obstacles=2,
            num_robots=1,
            tetherDist=2,
            num_leaders=1,
            num_target=1
        )
        
        parsed_obstacle_map = self.env.parse_obstacle_map(
            obstacle_map=obstacle_map
        )
        self.assertEqual(type(parsed_obstacle_map), np.ndarray)
        self.assertEqual(len(parsed_obstacle_map), 4)

    def test_parseAgentMap_listFormat(self):
        # Define a simple map
        obstacle_map = ["0000", "0101", "0001", "1000"]
        agent_map = ["0030", "0000", "0000", "0000"]
        target_map = ["0000", "0000", "0000", "0004"]

        # Initialize the environment
        self.env = SimpleGridEnv(
            obstacle_map=obstacle_map, 
            agent_map=agent_map, 
            target_map=target_map, 
            render_mode="human",
            rowSize=4,
            colSize=4,
            num_soft_obstacles=2,
            num_hard_obstacles=2,
            num_robots=1,
            tetherDist=2,
            num_leaders=1,
            num_target=1
        )
        
        parsed_agent_map = self.env.parse_agent_map(
            agent_map=agent_map
        )
        self.assertEqual(type(parsed_agent_map), list)
        self.assertEqual(len(parsed_agent_map), 1)

    def test_parseAgentMap_strFormat(self):
        # Define a simple map
        obstacle_map = "0000\n0101\n0001\n1000"
        agent_map = "0030\n0000\n0000\n0000"
        target_map = "0000\n0000\n0000\n0004"

        # Initialize the environment
        self.env = SimpleGridEnv(
            obstacle_map=obstacle_map, 
            agent_map=agent_map, 
            target_map=target_map, 
            render_mode="human",
            rowSize=4,
            colSize=4,
            num_soft_obstacles=2,
            num_hard_obstacles=2,
            num_robots=1,
            tetherDist=2,
            num_leaders=1,
            num_target=1
        )
        
        parsed_agent_map = self.env.parse_agent_map(
            agent_map=agent_map
        )
        self.assertEqual(type(parsed_agent_map), list)
        self.assertEqual(len(parsed_agent_map), 1)

    def test_parseTargetMap_listFormat(self):
        # Define a simple map
        obstacle_map = ["0000", "0101", "0001", "1000"]
        agent_map = ["0030", "0000", "0000", "0000"]
        target_map = ["0000", "0000", "0000", "0004"]

        # Initialize the environment
        self.env = SimpleGridEnv(
            obstacle_map=obstacle_map, 
            agent_map=agent_map, 
            target_map=target_map, 
            render_mode="human",
            rowSize=4,
            colSize=4,
            num_soft_obstacles=2,
            num_hard_obstacles=2,
            num_robots=1,
            tetherDist=2,
            num_leaders=1,
            num_target=1
        )
        
        parsed_target_map = self.env.parse_target_map(
            target_map=target_map
        )
        self.assertEqual(type(parsed_target_map), np.ndarray)
        self.assertEqual(len(parsed_target_map), 4)

    def test_parseTargetMap_strFormat(self):
        # Define a simple map
        obstacle_map = "0000\n0101\n0001\n1000"
        agent_map = "0030\n0000\n0000\n0000"
        target_map = "0000\n0000\n0000\n0004"

        # Initialize the environment
        self.env = SimpleGridEnv(
            obstacle_map=obstacle_map, 
            agent_map=agent_map, 
            target_map=target_map, 
            render_mode="human",
            rowSize=4,
            colSize=4,
            num_soft_obstacles=2,
            num_hard_obstacles=2,
            num_robots=1,
            tetherDist=2,
            num_leaders=1,
            num_target=1
        )
        
        parsed_target_map = self.env.parse_target_map(
            target_map=target_map
        )
        self.assertEqual(type(parsed_target_map), np.ndarray)
        self.assertEqual(len(parsed_target_map), 4)

    def test_initialize_with_predefined_maps(self):
        # Define a simple map
        obstacle_map = ["0000", "0101", "0001", "1000"]
        agent_map = ["0030", "0000", "0000", "0000"]
        target_map = ["0000", "0000", "0000", "0004"]

        # Initialize the environment with predefined maps
        self.env = SimpleGridEnv(
            obstacle_map=obstacle_map, 
            agent_map=agent_map, 
            target_map=target_map, 
            render_mode="human",
            rowSize=4,
            colSize=4,
            num_soft_obstacles=2,
            num_hard_obstacles=2,
            num_robots=1,
            tetherDist=2,
            num_leaders=1,
            num_target=1
        )
        
        self.assertEqual(self.env.nrow, 4)
        self.assertEqual(self.env.ncol, 4)
        self.assertEqual(self.env.obstacles.shape, (4, 4))
        self.assertEqual(self.env.targets.shape, (4, 4))
        self.assertEqual(len(self.env.agents), 1)

    def test_initialize_without_predefined_maps(self):
        # Initialize the environment without predefined maps
        self.env = SimpleGridEnv(
            obstacle_map=None, 
            agent_map=None, 
            target_map=None, 
            render_mode="human",
            rowSize=4,
            colSize=4,
            num_soft_obstacles=2,
            num_hard_obstacles=2,
            num_robots=1,
            tetherDist=2,
            num_leaders=1,
            num_target=1
        )
        
        self.assertEqual(self.env.nrow, 4)
        self.assertEqual(self.env.ncol, 4)
        self.assertEqual(self.env.obstacles.shape, (4, 4))
        self.assertEqual(self.env.targets.shape, (4, 4))
        self.assertEqual(len(self.env.agents), 1)

    def test_reset_with_predefined_maps(self):
        # Define a simple map
        obstacle_map = ["0000", "0101", "0001", "1000"]
        agent_map = ["0030", "0000", "0000", "0000"]
        target_map = ["0000", "0000", "0000", "0004"]

        # Initialize the environment with predefined maps
        self.env = SimpleGridEnv(
            obstacle_map=obstacle_map, 
            agent_map=agent_map, 
            target_map=target_map, 
            render_mode="human",
            rowSize=4,
            colSize=4,
            num_soft_obstacles=2,
            num_hard_obstacles=2,
            num_robots=1,
            tetherDist=2,
            num_leaders=1,
            num_target=1
        )
        
        initial_state = self.env.reset()
        self.assertEqual(self.env.nrow, 4)
        self.assertEqual(self.env.ncol, 4)
        self.assertEqual(self.env.obstacles.shape, (4, 4))
        self.assertEqual(self.env.targets.shape, (1, 2))
        self.assertEqual(len(self.env.agents), 1)
        self.assertEqual(type(initial_state), dict)
        self.assertIn('observation', initial_state)

    def test_reset_without_predefined_maps(self):
        # Initialize the environment without predefined maps
        self.env = SimpleGridEnv(
            obstacle_map=None, 
            agent_map=None, 
            target_map=None, 
            render_mode="human",
            rowSize=4,
            colSize=4,
            num_soft_obstacles=2,
            num_hard_obstacles=2,
            num_robots=1,
            tetherDist=2,
            num_leaders=1,
            num_target=1
        )
        
        initial_state = self.env.reset()
        self.assertEqual(self.env.nrow, 4)
        self.assertEqual(self.env.ncol, 4)
        self.assertEqual(self.env.obstacles.shape, (4, 4))
        self.assertEqual(self.env.targets.shape, (1, 2))
        self.assertEqual(len(self.env.agents), 1)
        self.assertEqual(type(initial_state), dict)
        self.assertIn('observation', initial_state)

    def test_parse_state_option_int(self):
        self.env = SimpleGridEnv(
            obstacle_map=None, 
            agent_map=None, 
            target_map=None, 
            render_mode="human",
            rowSize=4,
            colSize=4,
            num_soft_obstacles=2,
            num_hard_obstacles=2,
            num_robots=1,
            tetherDist=2,
            num_leaders=1,
            num_target=1
        )
        self.env.reset(seed=42)
        state = self.env.parse_state_option("start_loc", {"start_loc": 5})
        self.assertEqual(state, (1, 1))  # Assuming a 4x4 grid

    def test_parse_state_option_tuple(self):
        self.env = SimpleGridEnv(
            obstacle_map=None, 
            agent_map=None, 
            target_map=None, 
            render_mode="human",
            rowSize=4,
            colSize=4,
            num_soft_obstacles=2,
            num_hard_obstacles=2,
            num_robots=1,
            tetherDist=2,
            num_leaders=1,
            num_target=1
        )
        self.env.reset(seed=42)
        state = self.env.parse_state_option("goal_loc", {"goal_loc": (2, 2)})
        self.assertEqual(state, (2, 2))

    def test_parse_state_option_random(self):
        self.env = SimpleGridEnv(
            obstacle_map=None, 
            agent_map=None, 
            target_map=None, 
            render_mode="human",
            rowSize=4,
            colSize=4,
            num_soft_obstacles=2,
            num_hard_obstacles=2,
            num_robots=1,
            tetherDist=2,
            num_leaders=1,
            num_target=1
        )
        self.env.reset(seed=42)
        state = self.env.parse_state_option("start_loc", {})
        self.assertEqual(type(state), tuple)
        self.assertEqual(len(state), 2)

    def test_render_mode_rgb_array(self):
        # Define a simple map
        obstacle_map = ["0000", "0101", "0001", "1000"]
        agent_map = ["0030", "0000", "0000", "0000"]
        target_map = ["0000", "0000", "0000", "0004"]

        # Initialize the environment with render_mode="rgb_array"
        self.env = SimpleGridEnv(
            obstacle_map=obstacle_map, 
            agent_map=agent_map, 
            target_map=target_map, 
            render_mode="rgb_array",
            rowSize=4,
            colSize=4,
            num_soft_obstacles=2,
            num_hard_obstacles=2,
            num_robots=1,
            tetherDist=2,
            num_leaders=1,
            num_target=1
        )
        
        self.env.reset()
        img = self.env.render()
        self.assertEqual(type(img), np.ndarray)
        self.assertEqual(img.shape[2], 4)  # Check if the image has 4 channels (RGBA)
        print(f"Image shape: {img.shape}")
        print(f"Image: {img}")

    def test_render_mode_ansi(self):
        # Define a simple map
        obstacle_map = ["0000", "0101", "0001", "1000"]
        agent_map = ["0030", "0000", "0000", "0000"]
        target_map = ["0000", "0000", "0000", "0004"]

        # Initialize the environment with render_mode="ansi"
        self.env = SimpleGridEnv(
            obstacle_map=obstacle_map, 
            agent_map=agent_map, 
            target_map=target_map, 
            render_mode="ansi",
            rowSize=4,
            colSize=4,
            num_soft_obstacles=2,
            num_hard_obstacles=2,
            num_robots=1,
            tetherDist=2,
            num_leaders=1,
            num_target=1
        )
        
        self.env.reset()
        ansi_output = self.env.render()
        self.assertEqual(type(ansi_output), str)
        self.assertIn(",", ansi_output)  # Check if the output contains comma-separated values
        print(f"ANSI Output: {ansi_output}")

    def test_training_mode_no_reset(self):
        # Define a simple map
        obstacle_map = ["0000", "0101", "0001", "1000"]
        agent_map = ["0030", "0000", "0000", "0000"]
        target_map = ["0000", "0000", "0000", "0004"]

        # Initialize the environment
        self.env = SimpleGridEnv(
            obstacle_map=obstacle_map, 
            agent_map=agent_map, 
            target_map=target_map, 
            render_mode="human",
            rowSize=4,
            colSize=4,
            num_soft_obstacles=2,
            num_hard_obstacles=2,
            num_robots=1,
            tetherDist=2,
            num_leaders=1,
            num_target=1
        )
        self.env.reset()

        # Get the initial position of the agent
        initial_position = self.env.agents[0]['position']

        # Simulate an invalid move (e.g., out of bounds)
        actions = {0: ACTION_SPACE.UP.value}  # Move up, which is out of bounds
        obs, reward, done, truncated, info = self.env.step(actions, isTraining=True)

        # Verify that the environment does not reset
        self.assertEqual(self.env.agents[0]['position'], initial_position)
        self.assertFalse(done)  # The episode should not end
        self.assertIn('agent_positions', info)
        self.assertEqual(info['agent_positions'][0], initial_position)

if __name__ == '__main__':
    unittest.main()