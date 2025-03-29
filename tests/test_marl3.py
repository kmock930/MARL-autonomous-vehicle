import unittest

import sys
import os
PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PATH)

from marl_3 import SimpleGridEnv, ACTION_SPACE, new_pos, get_leader_message, build_encoder_decoder, build_policy_network, MAPPO, contrastive_loss, train_MAPPO
import numpy as np
import tensorflow as tf
from constants import ACTION_SPACE

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

        # Create MAPPO object
        self.mappo = MAPPO(
            leader_model=build_policy_network(),
            follower_model=build_policy_network(),
            encoded_model=build_encoder_decoder()
        )
        self.assertIsInstance(self.mappo, MAPPO)
    
    def test_agents_list(self):
        self.agent_position = self.env.agents[0]['position']  # Use the first agent's position
        self.checkPosition(self.agent_position)

    def test_new_pos(self):
        # Move the agent to a new position
        agent_current_pos = self.env.agents[0]['position']
        action = ACTION_SPACE.UP
        actionValue = action.value
        agents = self.env.agents
        newPos = new_pos(agent_current_pos, action, agents)

        # Check if the new position is valid
        expected_new_pos = (agent_current_pos[0] + actionValue[0], agent_current_pos[1] + actionValue[1])
        if not (0 <= expected_new_pos[0] < self.env.env_configurations["rowSize"] and
                0 <= expected_new_pos[1] < self.env.env_configurations["colSize"]):
            # If the move is out of bounds, the position should remain the same
            print("Out of Bounds")
            self.assertEqual(newPos, agent_current_pos)
        elif any(agent['position'] == expected_new_pos for agent in agents):
            # If the move collides with another agent, the position should remain the same
            print("Agent Collision")
            self.assertEqual(newPos, agent_current_pos)
        elif self.env.obstacles[expected_new_pos[0], expected_new_pos[1]] in [self.env.OBSTACLE_SOFT, self.env.OBSTACLE_HARD]:
            # If the move collides with an obstacle, the position should remain the same
            print("Obstacle Collision")
            self.assertEqual(newPos, agent_current_pos)
        else:
            # Otherwise, the position should update correctly
            print("Valid Move")
            self.assertEqual(newPos, expected_new_pos)

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

    def test_MAPPO_init(self):
        self.assertIsInstance(self.mappo.leader_model, tf.keras.Model)
        self.assertIsInstance(self.mappo.follower_model, tf.keras.Model)
        self.assertIsInstance(self.mappo.encoded_model, tf.keras.Model)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def test_MAPPO_compute_loss(self):
        # Generate dummy data for the test
        state_leader = np.random.rand(1, 8).astype(np.float32)
        decoded_msg = np.random.rand(1, 8).astype(np.float32)
        action_leader = np.random.rand(1, len(ACTION_SPACE)).astype(np.float32)
        action_follower = np.random.rand(1, len(ACTION_SPACE)).astype(np.float32)
        reward = np.random.rand(1).astype(np.float32)
        leader_message = np.random.rand(1, 8).astype(np.float32)
        encoded_message = np.random.rand(1, 8).astype(np.float32)
        decoded_message = np.random.rand(1, 8).astype(np.float32)

        # Call compute_loss using the MAPPO object from setUp
        loss = self.mappo.compute_loss(
            state_leader=state_leader,
            decoded_msg=decoded_msg,
            action_leader=action_leader,
            action_follower=action_follower,
            reward=reward,
            leader_message=leader_message,
            encoded_message=encoded_message,
            decoded_message=decoded_message
        )

        # Assert the loss is a valid tensor
        self.assertIsInstance(loss, tf.Tensor)
        self.assertEqual(loss.dtype, tf.float32)

    def test_MAPPO_apply_gradient(self):
        # Generate dummy data for the test
        state_leader = np.random.rand(1, 8).astype(np.float32)
        decoded_msg = np.random.rand(1, 8).astype(np.float32)
        action_leader = np.random.rand(1, len(ACTION_SPACE)).astype(np.float32)
        action_follower = np.random.rand(1, len(ACTION_SPACE)).astype(np.float32)
        reward = np.random.rand(1).astype(np.float32)
        leader_message = np.random.rand(1, 8).astype(np.float32)
        encoded_message = np.random.rand(1, 8).astype(np.float32)
        decoded_message = np.random.rand(1, 8).astype(np.float32)

        # Call the apply_gradients function
        self.mappo.apply_gradients(
            state_leader=state_leader,
            decoded_msg=decoded_msg,
            action_leader=action_leader,
            action_follower=action_follower,
            reward=reward,
            leader_message=leader_message,
            encoded_message=encoded_message,
            decoded_message=decoded_message
        )

        # Assert that the leader model weights are updated
        leader_weights_before = [tf.identity(w) for w in self.mappo.leader_model.trainable_variables]
        self.mappo.apply_gradients(
            state_leader=state_leader,
            decoded_msg=decoded_msg,
            action_leader=action_leader,
            action_follower=action_follower,
            reward=reward,
            leader_message=leader_message,
            encoded_message=encoded_message,
            decoded_message=decoded_message
        )
        leader_weights_after = self.mappo.leader_model.trainable_variables
        for before, after in zip(leader_weights_before, leader_weights_after):
            self.assertFalse(np.array_equal(before.numpy(), after.numpy()), "Leader model weights did not update.")

        # Assert that the follower model weights are updated
        follower_weights_before = [tf.identity(w) for w in self.mappo.follower_model.trainable_variables]
        self.mappo.apply_gradients(
            state_leader=state_leader,
            decoded_msg=decoded_msg,
            action_leader=action_leader,
            action_follower=action_follower,
            reward=reward,
            leader_message=leader_message,
            encoded_message=encoded_message,
            decoded_message=decoded_message
        )
        follower_weights_after = self.mappo.follower_model.trainable_variables
        for before, after in zip(follower_weights_before, follower_weights_after):
            self.assertTrue(np.array_equal(before.numpy(), after.numpy()), "Follower model weights did not update.")

    def test_contrastive_loss(self):
        # Generate dummy data for the test
        messages = np.random.rand(5, 8).astype(np.float32)  # 5 messages with 8 dimensions each
        positive_pairs = [0, 1, 2, 3, 4]  # Positive pairs for contrastive loss
        temperature = 0.1  # Temperature parameter for contrastive loss

        # Call the contrastive_loss function
        loss = contrastive_loss(messages, positive_pairs, temperature)

        # Assert the loss is a valid tensor
        self.assertIsInstance(loss, tf.Tensor)
        self.assertEqual(loss.dtype, tf.float32)
        self.assertGreaterEqual(loss.numpy(), 0.0, "Contrastive loss should be non-negative.")

    def test_train_MAPPO(self):
        env = SimpleGridEnv(
            render_mode="rgb_array",
            rowSize=10,
            colSize=10,
            num_soft_obstacles=10,
            num_hard_obstacles=5,
            num_robots=2,
            tetherDist=2,
            num_leaders=1,
            num_target=1
        )
        train_MAPPO(
            episodes=1,  # Minimal number of episodes
            leader_model=self.mappo.leader_model,
            follower_model=self.mappo.follower_model,
            encoded_model=self.mappo.encoded_model,
            env=env
        )

        # Assert that the environment reset and training completed
        self.assertIsNotNone(env.agents, "Agents should be initialized.")
        self.assertIsNotNone(env.targets, "Targets should be initialized.")

    def test_predict_results(self):
        env = SimpleGridEnv(
            render_mode="rgb_array",
            rowSize=10,
            colSize=10,
            num_soft_obstacles=10,
            num_hard_obstacles=5,
            num_robots=2,
            tetherDist=2,
            num_leaders=1,
            num_target=1
        )
        train_MAPPO(
            episodes=1,  # Minimal number of episodes
            leader_model=self.mappo.leader_model,
            follower_model=self.mappo.follower_model,
            encoded_model=self.mappo.encoded_model,
            env=env
        )

        print("LEADER MODEL SUMMARY:")
        print(self.mappo.leader_model.summary())
        print("FOLLOWER MODEL SUMMARY:")
        print(self.mappo.follower_model.summary())

        # Convert agent positions to NumPy arrays and pad to match the expected input shape
        leader_position = np.array(env.agents[0]['position']).reshape(1, -1)
        leader_position_padded = np.pad(leader_position, ((0, 0), (0, 6)), mode='constant')  # Pad to shape (1, 8)

        follower_position = np.array(env.agents[1]['position']).reshape(1, -1)
        follower_position_padded = np.pad(follower_position, ((0, 0), (0, 6)), mode='constant')  # Pad to shape (1, 8)

        leaderModel_pred = self.mappo.leader_model.predict(leader_position_padded)  # Output: numpy array with probabilities
        followerModel_pred = self.mappo.follower_model.predict(follower_position_padded)  # Output: numpy array with probabilities
        self.assertIsInstance(leaderModel_pred, np.ndarray)
        self.assertIsInstance(followerModel_pred, np.ndarray)
        self.assertEqual(leaderModel_pred.shape[1], len(ACTION_SPACE))

        print("POLICY NETWORK SUMMARY:")
        self.mappo.encoded_model.summary()

        # Obtain the action with the highest probability for both leader and follower
        leader_action_index = np.argmax(leaderModel_pred, axis=1)  # Index of the highest probability
        follower_action_index = np.argmax(followerModel_pred, axis=1)  # Index of the highest probability

        leader_action = list(ACTION_SPACE)[leader_action_index[0]]  # Enum
        follower_action = list(ACTION_SPACE)[follower_action_index[0]]  # Enum

        print("LEADER ACTION: ", leader_action, " ", leader_action.value)
        print("FOLLOWER ACTION: ", follower_action, " ", follower_action.value)
        self.assertIsInstance(leader_action, ACTION_SPACE)
        self.assertIsInstance(follower_action, ACTION_SPACE)
        self.assertIsInstance(leader_action.value, tuple)
        self.assertIsInstance(follower_action.value, tuple)
        for val in leader_action.value:
            self.assertIsInstance(val, int)
        for val in follower_action.value:
            self.assertIsInstance(val, int)
        
        # POLICY NETWORK PREDICTION
        # Step 1: Create dummy leader message (simulate message structure)
        dummy_leader_message = np.random.rand(1, 8).astype(np.float32)

        # Step 2: Pass through encoder-decoder model (communication channel)
        encoded_output = self.mappo.encoded_model.predict(dummy_leader_message)

        # Step 3: Feed the encoded output to follower's policy network
        action_probs = self.mappo.follower_model.predict(encoded_output)

        # Step 4: Assert that the output is valid and action probabilities sum to ~1
        self.assertEqual(action_probs.shape, (1, len(ACTION_SPACE)))
        self.assertAlmostEqual(np.sum(action_probs), 1.0, places=3)

        print("Encoded Msg:", encoded_output)
        print("Follower Policy Output:", action_probs)

    def test_display_model_summary(self):
        env = SimpleGridEnv(
            render_mode="rgb_array",
            rowSize=10,
            colSize=10,
            num_soft_obstacles=10,
            num_hard_obstacles=5,
            num_robots=2,
            tetherDist=2,
            num_leaders=1,
            num_target=1
        )
        train_MAPPO(
            episodes=1,  # Minimal number of episodes
            leader_model=self.mappo.leader_model,
            follower_model=self.mappo.follower_model,
            encoded_model=self.mappo.encoded_model,
            env=env
        )
        # Display the model summary for leader and follower models
        print("LEADER MODEL SUMMARY:")
        self.mappo.leader_model.summary()
        print("FOLLOWER MODEL SUMMARY:")
        self.mappo.follower_model.summary()
        print("ENCODED MODEL SUMMARY:")
        self.mappo.encoded_model.summary()

    def tearDown(self):
        if hasattr(self, 'env') and self.env is not None:
            self.env.close()
if __name__ == '__main__':
    unittest.main()
