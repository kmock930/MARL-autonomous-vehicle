import numpy as np
import sys
sys.path.append('gym-simplegrid')
import os
SIMPLEGRID_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'gym-simplegrid', 'gym_simplegrid', 'envs'))
sys.path.append(SIMPLEGRID_PATH)
from constants import ACTION_SPACE
from simple_grid import SimpleGridEnv
import tensorflow as tf

class Agent:
    _id_counter = 1  # Class-level counter for auto-assigning agent IDs

    agent_id: int # Unique identifier of the agent
    role: str # Role of the agent
    position: tuple[int, int] # Position of the agent
    action_space: list[tuple[int, int]] # Action space of the agent
    env: SimpleGridEnv # Environment of the agent
    encoded_message: np.ndarray # Encoded message from the encoder-decoder

    # Algorithms
    encoder_decoder: tf.keras.Model
    policy_network: tf.keras.Model

    def __init__(self, env, role: str, encoder_decoder: tf.keras.Model, policy_network: tf.keras.Model):
        """
        Generalized Agent class for interacting with the SimpleGridEnv.
        
        Parameters:
        - env: SimpleGridEnv instance
        - role: String indicator of the agent's role (leader or follower)
        - encoder_decoder: tf.keras.Model (trained)
        - policy_network: tf.keras.Model (trained)
        """
        self.env = env
        self.role = role
        self.action_space = [action.value for action in ACTION_SPACE]

        # Auto Assigning an Agent ID
        self.agent_id = Agent._id_counter
        Agent._id_counter += 1
                
        # Ensure agent_id is within bounds
        if self.agent_id >= len(self.env.agents):
            raise IndexError(f"Agent ID {self.agent_id} is out of bounds for the environment's agents list.")
        
        # Extract position from the environment
        self.position = self.env.agents[self.agent_id]['position']

        # Load models from Parameters
        self.encoder_decoder = encoder_decoder
        self.policy_network = policy_network
    
    def act(self, observation: np.ndarray) -> tuple[int, int]:
        """

        Args:
            observation (np.ndarray): partial obersevation of the environment

        Returns:
            tuple[int, int]: an action to be taken by the agent
        """
        # TODO: Not Implemented
        if (self.role == "leader"):
            self.speak()
        else:
            self.listen()
        # default: take a random action
        return self.action_space[np.random.choice(len(self.action_space))]
    
    def listen(self, message: np.ndarray) -> None:
        """

        Args:
            message (dict): message from the leader agent
        """
        if (self.role == "follower"):
            # should take encoder's message and return decoder's output
            self.encoded_message = message
            self.encoder_decoder.predict(message)
        else:
            raise ValueError("Leader cannot listen")
    
    def speak(self) -> list:
        """

        Returns:
            dict: message to be sent to the follower agent
        """
        if (self.role == "leader"):
            # should return encoder's output
            NotImplementedError("Leader agent does not speak")
        else:
            ValueError("Follower cannot speak")

    # util functions
    # LEADER
    def isLeader(self, pos):
        return self.env[pos] == self.env.AGENT and self.env.agents['role'] == 'leader'
    # FOLLOWER
    def isFollower(self, pos):
        return self.env[pos] == self.env.AGENT and self.env.agents['role'] == 'follower'
