import unittest
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
SIMPLEGRID_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'gym-simplegrid', 'gym_simplegrid', 'envs'))
print(SIMPLEGRID_PATH)
sys.path.append(SIMPLEGRID_PATH)
from simple_grid import SimpleGridEnv

class TestSimpleGridEnv(unittest.TestCase):
    def test_samplevalidStatexy(self):
        # Define a simple map
        obstacle_map = ["0000", "0101", "0001", "1000"]
        agent_map = ["0030", "0000", "0000", "0000"]
        target_map = ["0000", "0000", "0000", "0004"]

        # Initialize the environment
        env = SimpleGridEnv(
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
        
        valid_state:tuple = env.sample_valid_state_xy()
        self.assertEqual(type(valid_state), tuple)
        self.assertEqual(len(valid_state), 2)
    
    def test_parseObstacleMap_listFormat(self):
        # Define a simple map
        obstacle_map = ["0000", "0101", "0001", "1000"]
        agent_map = ["0030", "0000", "0000", "0000"]
        target_map = ["0000", "0000", "0000", "0004"]

        # Initialize the environment
        env = SimpleGridEnv(
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
        
        parsed_obstacle_map = env.parse_obstacle_map(
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
        env = SimpleGridEnv(
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
        
        parsed_obstacle_map = env.parse_obstacle_map(
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
        env = SimpleGridEnv(
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
        
        parsed_agent_map = env.parse_agent_map(
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
        env = SimpleGridEnv(
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
        
        parsed_agent_map = env.parse_agent_map(
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
        env = SimpleGridEnv(
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
        
        parsed_target_map = env.parse_target_map(
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
        env = SimpleGridEnv(
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
        
        parsed_target_map = env.parse_target_map(
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
        env = SimpleGridEnv(
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
        
        self.assertEqual(env.nrow, 4)
        self.assertEqual(env.ncol, 4)
        self.assertEqual(env.obstacles.shape, (4, 4))
        self.assertEqual(env.targets.shape, (4, 4))
        self.assertEqual(len(env.agents), 1)

    def test_initialize_without_predefined_maps(self):
        # Initialize the environment without predefined maps
        env = SimpleGridEnv(
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
        
        self.assertEqual(env.nrow, 4)
        self.assertEqual(env.ncol, 4)
        self.assertEqual(env.obstacles.shape, (4, 4))
        self.assertEqual(env.targets.shape, (4, 4))
        self.assertEqual(len(env.agents), 1)

    def test_reset_with_predefined_maps(self):
        # Define a simple map
        obstacle_map = ["0000", "0101", "0001", "1000"]
        agent_map = ["0030", "0000", "0000", "0000"]
        target_map = ["0000", "0000", "0000", "0004"]

        # Initialize the environment with predefined maps
        env = SimpleGridEnv(
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
        
        initial_state = env.reset()
        self.assertEqual(env.nrow, 4)
        self.assertEqual(env.ncol, 4)
        self.assertEqual(env.obstacles.shape, (4, 4))
        self.assertEqual(env.targets.shape, (1, 2))
        self.assertEqual(len(env.agents), 1)
        self.assertEqual(type(initial_state), dict)
        self.assertIn('observation', initial_state)

    def test_reset_without_predefined_maps(self):
        # Initialize the environment without predefined maps
        env = SimpleGridEnv(
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
        
        initial_state = env.reset()
        self.assertEqual(env.nrow, 4)
        self.assertEqual(env.ncol, 4)
        self.assertEqual(env.obstacles.shape, (4, 4))
        self.assertEqual(env.targets.shape, (1, 2))
        self.assertEqual(len(env.agents), 1)
        self.assertEqual(type(initial_state), dict)
        self.assertIn('observation', initial_state)

    def test_parse_state_option_int(self):
        env = SimpleGridEnv(
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
        env.reset(seed=42)
        state = env.parse_state_option("start_loc", {"start_loc": 5})
        self.assertEqual(state, (1, 1))  # Assuming a 4x4 grid

    def test_parse_state_option_tuple(self):
        env = SimpleGridEnv(
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
        env.reset(seed=42)
        state = env.parse_state_option("goal_loc", {"goal_loc": (2, 2)})
        self.assertEqual(state, (2, 2))

    def test_parse_state_option_random(self):
        env = SimpleGridEnv(
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
        env.reset(seed=42)
        state = env.parse_state_option("start_loc", {})
        self.assertEqual(type(state), tuple)
        self.assertEqual(len(state), 2)

if __name__ == '__main__':
    unittest.main()