import unittest

import sys
import os
PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PATH)

from marl_3 import SimpleGridEnv, ACTION_SPACE, new_pos, get_leader_message, build_encoder_decoder, build_policy_network
import numpy as np
import tensorflow as tf

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

        # Check if the new position is valid
        if not (0 <= agent_current_pos[0] + action[0] < self.env.env_configurations["rowSize"] and
                0 <= agent_current_pos[1] + action[1] < self.env.env_configurations["colSize"]):
            # If the move is out of bounds, the position should remain the same
            print("Out of Bounds")
            self.assertEqual(newPos, agent_current_pos)
        elif any(agent['position'] == (agent_current_pos[0] + action[0], agent_current_pos[1] + action[1]) for agent in agents):
            # If the move collides with another agent, the position should remain the same
            print("Agent Collision")
            self.assertEqual(newPos, agent_current_pos)
        elif self.env.obstacles[agent_current_pos[0] + action[0], agent_current_pos[1] + action[1]] in [self.env.OBSTACLE_SOFT, self.env.OBSTACLE_HARD]:
            # If the move collides with an obstacle, the position should remain the same
            print("Obstacle Collision")
            self.assertEqual(newPos, agent_current_pos)
        else:
            # Otherwise, the position should update correctly
            print("Valid Move")
            self.assertEqual(newPos, (agent_current_pos[0] + action[0], agent_current_pos[1] + action[1]))

    def test_get_leader_message(self):
        # Get the position of the leader agent
        leader_pos = next(agent['position'] for agent in self.env.agents if agent.get('role') == 'leader')
        self.checkPosition(leader_pos)

        # Generate the leader message
        message = get_leader_message(leader_pos, self.env)
        print("LEADER MESSAGE: ", message)

        # Check the structure of the message
        self.assertIsInstance(message, list)
        self.assertEqual(len(message), 6)  # Ensure the message has 6 elements
        self.assertTrue(all(isinstance(value, (int, float)) for value in message[:5]))  # Check numeric values
        self.assertIn(message[5], [0, 1])  # Ensure path_blocked is 0 or 1

        # Additional checks for specific values
        xg, yg, obs_dist, follower_visibility, follower_dist, path_blocked = message
        if follower_visibility == 1:
            self.assertGreaterEqual(follower_dist, 0)  # Ensure valid follower distance
        if obs_dist != -1:
            self.assertGreaterEqual(obs_dist, 0)  # Ensure valid obstacle distance

    def test_build_encoder_decoder(self):
        # Build the encoder-decoder model
        model = build_encoder_decoder()

        # Check if the model is an instance of tf.keras.Model
        self.assertIsInstance(model, tf.keras.Model)

        # Check the input and output shapes
        input_shape = model.input_shape
        output_shape = model.output_shape
        self.assertEqual(input_shape, (None, 8))  # Input shape should match the expected input size
        self.assertEqual(output_shape, (None, 8))  # Output shape should match the expected output size

        # Check the activation function of the output layer
        output_activation = model.layers[-1].activation.__name__
        self.assertEqual(output_activation, "linear")  # Ensure the output layer uses linear activation

        # Test a forward pass with dummy data
        dummy_input = np.random.rand(1, 8).astype(np.float32)
        output = model.predict(dummy_input)
        self.assertEqual(output.shape, (1, 8))  # Ensure the output shape matches the input shape

    def test_build_policy_network(self):
        # Build the policy network model
        model = build_policy_network()

        # Check if the model is an instance of tf.keras.Model
        self.assertIsInstance(model, tf.keras.Model)

        # Check the input and output shapes
        input_shape = model.input_shape
        output_shape = model.output_shape
        self.assertEqual(input_shape, (None, 8))  # Input shape should match the expected input size
        self.assertEqual(output_shape, (None, len(ACTION_SPACE)))  # Output shape should match the number of actions

        # Check the activation function of the output layer
        output_activation = model.layers[-1].activation.__name__
        self.assertEqual(output_activation, "softmax")  # Ensure the output layer uses softmax activation

        # Check the activation functions of the hidden layers
        hidden_activations = [layer.activation.__name__ for layer in model.layers if hasattr(layer, 'activation') and layer.activation]
        for activation in hidden_activations[:-1]:  # Exclude the output layer
            self.assertIn(activation, ["relu", "tanh", "sigmoid"])  # Ensure hidden layers use valid activation functions

        # Test a forward pass with dummy data
        dummy_input = np.random.rand(1, 8).astype(np.float32)
        output = model.predict(dummy_input)
        self.assertEqual(output.shape, (1, len(ACTION_SPACE)))  # Ensure the output shape matches the expected output size

if __name__ == '__main__':
    unittest.main()
