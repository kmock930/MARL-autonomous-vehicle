import unittest

import sys
import os
PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PATH)

from marl_3 import SimpleGridEnv, ACTION_SPACE, new_pos
import numpy as np

class TestMoveAgent(unittest.TestCase):
    def checkPosition(self, coord):
        self.assertIsInstance(coord, tuple)
        self.assertEqual(len(coord), 2)
        self.assertTrue(all(isinstance(pos, int) for pos in coord))
        self.assertTrue(0 <= coord[0] < self.env.env_configurations["rowSize"])
        self.assertTrue(0 <= coord[1] < self.env.env_configurations["colSize"])

    def setUp(self):
        # Initialize the SimpleGridEnv
        self.env = SimpleGridEnv(
            render_mode=None,
            rowSize=10,
            colSize=10,
            num_soft_obstacles=10,
            num_hard_obstacles=5,
            num_robots=2,
            tetherDist=2,
            num_leaders=1,
            num_target=1
        )
        self.env.reset()
    
    def test_agents_list(self):
        self.agent_position = self.env.agents[0]['position']  # Use the first agent's position
        self.checkPosition(self.agent_position)

    def test_new_pos(self):
        # Move the agent to a new position
        agent_current_pos = self.env.agents[0]['position']
        action = ACTION_SPACE.UP.value
        agents = self.env.agents
        newPos = new_pos(agent_current_pos, action, agents)
        self.checkPosition(newPos)
        self.assertEqual(newPos, (agent_current_pos[0] + action[0], agent_current_pos[1] + action[1]))


if __name__ == '__main__':
    unittest.main()
