import numpy as np
import sys
sys.path.append('gym-simplegrid')
from constants import ACTION_SPACE

class Agent:
    role: str # Role of the agent
    position: tuple[int, int] # Position of the agent
    action_space: tuple[int, int] # Action space of the agent

    def __init__(self, role: str, position: tuple[int, int]):
        self.role = role
        self.position = position
        self.action_space = ACTION_SPACE
    
    def act(self, observation: np.ndarray) -> tuple[int, int]:
        """_summary_

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
        return ACTION_SPACE[8] # Stay
    
    def listen(self, message: dict) -> None:
        """_summary_

        Args:
            message (dict): message from the leader agent
        """
        if (self.role == "follower"):
            NotImplementedError("Follower agent does not listen")
        else:
            ValueError("Leader cannot listen")
    
    def speak(self) -> dict:
        """_summary_

        Returns:
            dict: message to be sent to the follower agent
        """
        if (self.role == "leader"):
            NotImplementedError("Leader agent does not speak")
        else:
            ValueError("Follower cannot speak")