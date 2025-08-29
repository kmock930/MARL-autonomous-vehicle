# This is the new implementation
# where the leader message size = 8

"""marl_6.py
"""

import tensorflow as tf
import matplotlib.pyplot as plt

import numpy as np
from constants import *
# Import the Env
import sys
import os
SIMPLEGRID_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'gym-simplegrid', 'gym_simplegrid', 'envs'))
sys.path.append(SIMPLEGRID_PATH)
from simple_grid import SimpleGridEnv
from agent import Agent  # Import the Agent class
from dru import DRU_DIAL
import pandas as pd
import datetime

FREE: int = 0
OBSTACLE_SOFT: int = 1
OBSTACLE_HARD: int = 2
AGENT: int = 3
TARGET: int = 4

# Define LEADER and FOLLOWER constants for role-based checks
LEADER = "leader"
FOLLOWER = "follower"

# Initialize the environment
env = SimpleGridEnv(
    render_mode="rgb_array", # numpy array representation
    rowSize=10,
    colSize=10,
    num_soft_obstacles=10,
    num_hard_obstacles=5,
    num_robots=2,
    tetherDist=2,
    num_leaders=1,
    num_target=1
)

#window: 3X3
def get_agent_observation(pos: tuple[int, int], env: SimpleGridEnv, agent_mode: str, visible_dist: int = 1, returnOwn: bool = False) -> list:
    """
    Generate an observation for the agents based on its position and the environment.
    leader and follower will use this to get their action space. Follower will have 
    additional information about the leader through leader's. 

    Args:
        pos (tuple[int, int]): The position of the leader agent.
        env (SimpleGridEnv): The environment instance.
        agent_mode (str): The mode of the agent (either 'leader' or 'follower').
        visible_dist (int): The distance within which the agent can observe its surroundings, default to be 1 block away.

    Returns:
        list: Agent's observation containing information about the environment around the agent.

    """
    x, y = pos  # Unpack the position
    agent_visibility = 0  # Leader cannot observe the follower initially
    agent_dist, obs_dist, counter = -1, -1, 0
    obstacles_pos, distances = [], []
    path_blocked = 0  # Path is not blocked initially
    action_dx, action_dy = 0, 0  # Default action
    curr_state = []

    for dx in range(-visible_dist, visible_dist + 1):
        for dy in range(-visible_dist, visible_dist + 1):
            nx, ny = x + dx, y + dy
            if 0 <= nx < env.env_configurations["rowSize"] and 0 <= ny < env.env_configurations["colSize"]: # valid coordinates
                counter += 1
                ##########################
                # Construct a list of 8 or 9 integers of what the agent sees
                # by getting the encoding values of each neighboring cell
                cell_value = env.obstacles[nx, ny]
                if cell_value in [0, None]:
                    if all((nx, ny) != agent["position"] for agent in env.agents) and \
                        0 <= nx < env.targets.shape[0] and 0 <= ny < env.targets.shape[1] and \
                        env.targets[nx, ny] != env.TARGET:
                            cell_value = FREE  # Free space
                    
                    # Ensure targets array has the correct dimensions
                    if env.targets.shape == (env.env_configurations["rowSize"], env.env_configurations["colSize"]):
                        # Relative position to the goal
                        if env.targets[nx, ny] == env.TARGET:
                            cell_value = TARGET  # Target space
                # Append to curr_state list by checking the args whether or not to return its own current position
                if not returnOwn and (dx, dy) == (0, 0):
                    pass
                else:
                    curr_state.append(cell_value)
                ##########################
                if agent_mode == 'leader':
                    # Agents can see each other
                    if any(agent['position'] == (nx, ny) and agent.get('role') == 'follower' for agent in env.agents):
                        agent_visibility = 1
                        agent_dist = np.floor(np.sqrt((x - nx) ** 2 + (y - ny) ** 2)) # Round down for diagonal distances
                elif agent_mode == 'follower':                    
                    if any(agent['position'] == (nx, ny) and agent.get('role') == 'leader' for agent in env.agents):
                        agent_visibility = 1
                        agent_dist = np.floor(np.sqrt((x - nx) ** 2 + (y - ny) ** 2)) # Round down for diagonal distances
                
                # Nearest obstacle
                if env.obstacles[nx, ny] in [env.OBSTACLE_HARD]:
                    obstacles_pos.append((nx, ny))
                    dist = np.floor(np.sqrt((x - nx) ** 2 + (y - ny) ** 2)) # Round down for diagonal distances
                    distances.append(dist)
            else:
                # If it is not a valid location, 
                # pad it with a placeholder value to keep ordering of the list the same across agents
                curr_state.append(-1)
    
    if len(distances) >= 0:
        obs_dist = min(distances) if len(distances) > 0 else 0
    
    if len(obstacles_pos) == counter:
        path_blocked = 1

    # Flatten everything into a 1D list
    final_obs = curr_state + [obs_dist, agent_visibility, agent_dist, path_blocked]
    print(f"Final Obs: {final_obs}")
    return final_obs

# Modify the `new_pos` function to check roles using the Agent class
def new_pos(agent_position: tuple[int, int], action: ACTION_SPACE, agents: list):
    x, y = agent_position
    dx, dy = action.value

    new_pos = (x + dx, y + dy)

    # Check if the new position is within the grid
    if not (0 <= new_pos[0] < env.env_configurations["rowSize"] and 0 <= new_pos[1] < env.env_configurations["colSize"]):
        print("Out of Bounds")  # Debugging message
        return agent_position  # Reverse because Invalid Move

    # Check if the new position is occupied by another agent
    for agent in agents:
        if agent["position"] == new_pos:
            print(f"Agent at {agent_position} collided with agent at {new_pos}")
            return agent_position  # Reverse because Invalid Move


    # Check if the new position is a hard obstacle
    if env.obstacles[new_pos[0], new_pos[1]] in [OBSTACLE_HARD]:
        print("Obstacle Collision")  # Debugging message
        return agent_position  # Reverse because Invalid Move

    print("Valid Move")  # Debugging message
    return new_pos


def plot_gradients(grads, label_prefix):
    grad_norms = [tf.norm(g).numpy() if g is not None else 0 for g in grads]
    names = [f"{label_prefix}_{i}" for i in range(len(grad_norms))]
    plt.bar(names, grad_norms, alpha=0.7, label=label_prefix)
    plt.xticks(rotation=90)

# MLP MAPPO
def leader_policy_network():
    input_layer = tf.keras.layers.Input(shape=(LEADER_OBS_SIZE,))
    reshaped = tf.keras.layers.Reshape((1, LEADER_OBS_SIZE))(input_layer)

    # hidden layers
    x = tf.keras.layers.Dense(64, activation="relu")(reshaped)
    x = tf.keras.layers.Dense(64, activation="relu")(x)

    # Action head
    action_output = tf.keras.layers.Dense(len(ACTION_SPACE), activation="softmax", name="action")(x)
    
    # Communication head (message generation)
    message_output = tf.keras.layers.Dense(MESSAGE_SIZE, activation="linear", name="message")(x)

    # output layer
    output_layer = {"action": action_output, "message": message_output}
    return tf.keras.models.Model(input_layer, output_layer)

def follower_policy_network():
    # Follower takes both observation + received message input as the input
    input_layer = tf.keras.Input(shape=(FOLLOWER_OBS_SIZE,))
    # Hidden layers remain unchanged
    x = tf.keras.layers.Dense(64, activation="relu")(input_layer)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    # Output layer: use softmax activation to output probabilities over the ACTION_SPACE.
    output_layer = tf.keras.layers.Dense(len(ACTION_SPACE), activation="softmax")(x)
    # Build and return the model
    return tf.keras.Model(inputs=input_layer, outputs=output_layer)

leader_policy = leader_policy_network() 
#Change the message size however you want but it should match with the follower policy network
follower_policy = follower_policy_network() 
#Change the message size however you want but it should match with the leader policy network

def build_critic_network():
    input_layer = tf.keras.Input(shape=(LEADER_OBS_SIZE,))
    x = tf.keras.layers.Dense(64, activation="relu")(input_layer)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    output_layer = tf.keras.layers.Dense(1)(x)  # Scalar value
    return tf.keras.Model(inputs=input_layer, outputs=output_layer)

critic_model = build_critic_network()

class MAPPO:
    def __init__(self, leader_model, follower_model, critic_model,lr=0.001):
        self.leader_model = leader_model
        self.follower_model = follower_model
        self.critic_model = critic_model 
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    def compute_loss(self, state_leader, state_follower, reward, leader_message, hyperparams: dict = None):
        # Hyperparameters
        entropy_bonus_weight = 0.01  # Default value
        if hyperparams:
            entropy_bonus_weight = hyperparams.get('entropy_bonus_weight', entropy_bonus_weight)

        # Convert inputs to tensors
        state_leader = tf.convert_to_tensor(state_leader, dtype=tf.float32)
        state_leader = tf.reshape(state_leader, (1, -1))

        # Compute Advantage (A = R + γV(s') - V(s))
        # Correctly call the critic model to predict the value
        value = self.critic_model(state_leader)[0, 0] #add state follower
        advantage = reward - value  # TD error as Advantage Estimate
        print("loss", advantage)

        # Policy Gradient Loss (A2C)
        leader_pred = self.leader_model(tf.reshape(state_leader, (1, -1)))
        action_prob_leader, leader_message_pred = leader_pred['action'], leader_pred['message']
        # Combine leader_message with follower's observations to create the input for the follower model
        follower_input = tf.expand_dims(tf.concat([state_follower, leader_message], axis=0), axis=0)
        action_prob_follower = self.follower_model(follower_input)
        policy_loss = -tf.reduce_mean(advantage * tf.math.log(action_prob_leader + 1e-8))-tf.reduce_mean(advantage * tf.math.log(action_prob_follower + 1e-8))
        print('Policy Gradient Loss', policy_loss)

        # Entropy Bonus for Exploration
        entropy_bonus = -tf.reduce_mean(action_prob_leader * tf.math.log(action_prob_leader + 1e-8))
        print('Entropy Bonus', entropy_bonus)

        # Final loss function
        total_loss = policy_loss + entropy_bonus_weight * entropy_bonus
        print('Total Loss', total_loss)

        return total_loss, [policy_loss, entropy_bonus]

    def apply_gradients(self, state_leader, reward, leader_message):
        with tf.GradientTape() as tape:
            loss, separate_losses = self.compute_loss(
                state_leader=state_leader,
                reward=reward,
                leader_message=leader_message
            )
        grads = tape.gradient(loss, self.leader_model.trainable_variables + self.follower_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.leader_model.trainable_variables + self.follower_model.trainable_variables))

def train_MAPPO(episodes, leader_model, follower_model, env, critic_model, hyperparams: dict = None, 
                algorithm="MAPPO", tether_tolerate_count=TETHER_TOLERATE_COUNT, isTraining=True):
    print("Starting training...")
    # Logging
    episode_rewards = []
    episode_losses = []
    episode_logs = []  # To store detailed logs for each episode
    gradient_changes = []
    episode_accuracies = []

    # Hyperparameters
    lr = 0.001  # Default learning rate
    max_step_per_episode = 100  # Default max steps per episode
    max_episodes = 100  # Default max episodes
    if hyperparams:
        lr = hyperparams.get('lr', lr)
        max_step_per_episode = hyperparams.get('max_steps', max_step_per_episode)
        max_episodes = hyperparams.get('max_episodes', max_episodes)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    total_rewards = []
    success_rate = 0

    episodes = episodes if (episodes is not None or episodes > 0) else max_episodes
    for episode in range(episodes):
        print(f"\nEpisode {episode + 1}/{episodes}")
        # Reset the environment
        obs = env.reset()
        leader_pos = env.leaders[0]['position']
        follower_pos = env.followers[0]['position']

        # Ensure there are targets in the environment
        target_positions = np.argwhere(env.targets == env.TARGET)
        target_pos = target_positions[0] if len(target_positions) > 0 else None
        
        episode_reset = False
        total_reward = 0
        leader_path = [leader_pos]
        follower_path = [follower_pos]

        reward = 0
        tether_violated = 0
        collisions = 0
        distances = []

        episode_entropy_bonus = []  # Initialize entropy_bonus to avoid UnboundLocalError
        episode_policy_loss = []  # Initialize loss to avoid UnboundLocalError
        episode_total_loss = []
        #Step takes should be outside the loop
        steps_taken = 0
        for step in range(max_step_per_episode):  # Limit the number of steps per episode
            # Initialize counters
            communication_count = 0
            print(f"Step {step + 1}/{max_step_per_episode}")
            # Leader generates a message and takes an action
            print("leader")
            leader_obs = get_agent_observation(leader_pos, env,'leader')
            state_leader = leader_obs  # Make a copy of the observation
            leader_obs_size = len(state_leader) # length of final_obs array
            
            
            leader_pred = leader_model(tf.convert_to_tensor(leader_obs, dtype=tf.float32)[None, :])
            leader_action_prob, leader_message = leader_pred.get('action', (None, None)),leader_pred.get('message', None)
            leader_action = list(ACTION_SPACE)[int(tf.argmax(tf.stop_gradient(leader_action_prob)[0, 0]).numpy())]

            communication_count += 1
            
            # Update leader position using the step method
            _, _, _, _, info = env.step(
                actions={0: leader_action.value},
                isTraining=True
            )
            print(f"Leader Position: {leader_pos}, Leader Action: {leader_action}")
            # new_leader_pos = new_pos(leader_pos, leader_action, env.agents)  # Pass the agents list
            new_leader_pos = info['agent_positions'][0]
            print(f"New Leader Position: {new_leader_pos}")
            
            # NOTE: IF WE USE THE NEW_POS FUNCTION INSTEAD OF THE STEP FUNCTION THEN REMEMBER TO UPDATE THE 
            # VALUE OF ORIGINAL CELL OF THE AGENT AFTER IT LEFT IT (OR KEEP IT SAME INCASE IT'S NEW POSITION
            # IS SAME AS THE ORIGINAL ONE. THE LOGIC IS AT THE BOTTOM OF THE STEP FUNCTION OF SIMPLE GRID)
            # ALSO UPDATE THE REWARD CALCULATION IN BOTH STEP AND IN THIS OUTER LOOP BELOW AND THEN CHOOSE
            # THE BEST METHOD. I.E. EITHER STEP CALL OR NEW_POS CALL.
            # CURRENTLY IN THE STAY FUNCTION THE REWARD CALCULATION IS WRONG. THE AGENT'S TARGET POSITION IS 
            # IS FED TO CALCUALTE REWARD NOT THE FINAL POSITION. BUT WE ALLOW STEP TO MAKE ONLY VALID MOVES
            # SO INVALID MOVES ARE NEVER PENALIZED AND WE GET ONLY -3 FOR 1 STAY

            # Checking if leader's action lead it to collide into the follower.
            # Now we check for only collision hjere because if leader collides into the follower
            # we will reverse the leader's actions before follower takes it's actions. We don't want
            # any collisions to remain. i.e. if collision happens then reverse it immdeiately.
            # If leader bumps into obstacles it is okay. Not collide into another agent.
            
            # Compute distance
            distance = np.floor(np.sqrt((new_leader_pos[0] - follower_pos[0])**2 + (new_leader_pos[1] - follower_pos[1])**2)) # Round down for diagonal distances
            distances.append(distance)

            if distance < 1: 
                # Collision
                collisions += 1
                print(f"Episode {episode+1}: Collision with another agent occurred. Reversing move...")
                # reward += REWARDS.CRASH.value  # add this line here
                new_leader_pos = leader_pos  # Reverse leader move

            # Getting the leader's message and passing it through DIAL
            if algorithm == "MAPPO":
                leader_message = leader_message
            elif algorithm == "DIAL":
                dru = DRU_DIAL(sigma=SIGMA, comm_narrow=True, hard=False)  # instantiate once, not every step
                leader_message = dru(leader_message, train_mode=isTraining)
            
            # Flatten leader message
            leader_message = tf.reshape(leader_message, (-1,))

            # Follower takes an action based on the decoded message
            print("follower")
            follower_obs = get_agent_observation(follower_pos, env,'follower')
            state_follower = follower_obs
            follower_obs_size = len(state_follower)  # length of final_obs array

            # Throw Exception in case if they are STILL not aligned
            if leader_obs_size != follower_obs_size: 
                raise ValueError(f"Leader observation size {leader_obs_size} does not match follower observation size {follower_obs_size}.")

            # Concatenate the Follower's observation with the Leader's message
            print(f"Leader's Message: {leader_message}")
            # Follower input (as TF tensor)
            follower_obs_tensor = tf.convert_to_tensor(follower_obs, dtype=tf.float32)
            combined_input = tf.concat([follower_obs_tensor, leader_message], axis=0)
            # Reshape to add batch dimension: (features,) -> (1, features)
            combined_input = combined_input[None, :]
            print(f"Follower's Input Shape: {combined_input.shape}")

            follower_action_probs = follower_model(combined_input)
            follower_action = list(ACTION_SPACE)[int(tf.argmax(tf.stop_gradient(follower_action_probs)[0]).numpy())]
            # Update follower position using the step method
            _, _, _, _, info = env.step(
                actions={1: follower_action.value},
                isTraining=True
            )
            print(f"Follower Position: {follower_pos}, Follower Action: {follower_action}")
            # new_follower_pos = new_pos(follower_pos, follower_action, env.agents)  # Pass the agents list
            new_follower_pos = info['agent_positions'][1]

            print(f"New Follower Position: {new_follower_pos}")
            
            
            # Compute distance
            distance = np.floor(np.sqrt((new_leader_pos[0] - new_follower_pos[0])**2 + (new_leader_pos[1] - new_follower_pos[1])**2)) # Round down for diagonal distances
            distances.append(distance)

            x_l, y_l = new_leader_pos
            x_f, y_f = new_follower_pos

            # Use tetherDist from the environment configuration
            tether_limit = env.env_configurations["tetherDist"]
            if distance > tether_limit: # or distance < 1:
                tether_violated += 1
                print(f"Episode {episode+1}: Tether constraint violated (Distance: {distance:.2f}, Tether Limit: {tether_limit}).")
            elif distance < 1:
                # Checking if follower's action lead it to collide into the leader
                collisions += 1
                print(f"Episode {episode+1}: Collision with another agent occurred. Reversing move...")
                
                new_follower_pos = follower_pos  # Reverse follower move
            elif env.obstacles[x_l, y_l] == OBSTACLE_HARD or env.obstacles[x_f, y_f] == OBSTACLE_HARD:
                print(f"Episode {episode+1}: Hard obstacle encountered. Reversing move...")
                new_leader_pos = leader_pos  # Reverse leader move
                new_follower_pos = follower_pos  # Reverse follower move

            # Update the path and position
            for agent in env.agents:
                if agent['position'] == follower_pos:
                    agent['position'] = new_follower_pos
                    break
            follower_pos = new_follower_pos
            follower_path.append(follower_pos)

            for agent in env.agents:
                if agent['position'] == leader_pos:
                    agent['position'] = new_leader_pos
                    break
            leader_pos = new_leader_pos
            leader_path.append(leader_pos)

            # Compute reward
            if (0 <= x_l < env.targets.shape[0] and 0 <= y_l < env.targets.shape[1] and env.targets[x_l, y_l] == TARGET) or \
               (0 <= x_f < env.targets.shape[0] and 0 <= y_f < env.targets.shape[1] and env.targets[x_f, y_f] == TARGET):
                reward += REWARDS.TARGET.value
                total_reward += reward  # Ensure the reward is added before exiting
                print(f"Target reached! Episode ends with reward: {total_reward}")
                episode_reset = True
                break
            elif (0 <= x_l < env.obstacles.shape[0] and 0 <= y_l < env.obstacles.shape[1] and env.obstacles[x_l, y_l] == OBSTACLE_SOFT) or \
                 (0 <= x_f < env.obstacles.shape[0] and 0 <= y_f < env.obstacles.shape[1] and env.obstacles[x_f, y_f] == OBSTACLE_SOFT):
                reward += REWARDS.SOFT_OBSTACLE.value
            elif not (0 <= x_l < env.obstacles.shape[0] and 0 <= y_l < env.obstacles.shape[1]) or \
                 not (0 <= x_f < env.obstacles.shape[0] and 0 <= y_f < env.obstacles.shape[1]):
                reward += REWARDS.WALL.value  # Penalty for out-of-bound situations
            elif env.obstacles[x_l, y_l] == OBSTACLE_HARD or env.obstacles[x_f, y_f] == OBSTACLE_HARD:
                reward += REWARDS.HARD_OBSTACLE.value  # Penalty for crashing into hard obstacles
            elif any(agent['position'] == new_leader_pos for agent in env.agents if agent['position'] != leader_pos) or \
                 any(agent['position'] == new_follower_pos for agent in env.agents if agent['position'] != follower_pos):
                reward += REWARDS.CRASH.value  # Penalty for crashing onto another agent
            elif distance > env.env_configurations["tetherDist"]:
                reward += REWARDS.OUT_OF_TETHER.value * tether_violated  # Penalty for being out of tether range)
            elif (x_l == new_leader_pos[0] and y_l == new_leader_pos[1]) or \
                 (x_f == new_follower_pos[0] and y_f == new_follower_pos[1]):
                reward += REWARDS.STAY.value # Penalty for staying in the same position
            total_reward += reward

            if (tether_violated > tether_tolerate_count):
                break

            mappo_model = MAPPO(leader_model, follower_model, critic_model, lr)
            print("mappo")
            with tf.GradientTape() as tape:
                loss, separate_losses = mappo_model.compute_loss(
                    state_leader=tf.convert_to_tensor(state_leader, dtype=tf.float32),
                    state_follower=tf.convert_to_tensor(state_follower, dtype=tf.float32),
                    reward=reward,
                    leader_message=leader_message,   # must be tensor
                    hyperparams=hyperparams
                )
                policy_loss, entropy_bonus = separate_losses[0], separate_losses[1]
                episode_entropy_bonus.append(entropy_bonus)
                episode_policy_loss.append(policy_loss)
                episode_total_loss.append(loss)

            # Update Policy
            print("Update Policy")
            grads = tape.gradient(loss, leader_model.trainable_variables + follower_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, leader_model.trainable_variables + follower_model.trainable_variables))

            gradient_norm = tf.linalg.norm(grads[0]).numpy()  
            gradient_changes.append(gradient_norm)

            print(f"Step {step + 1}: Leader Action: {leader_action}, Follower Action: {follower_action}, Reward: {reward:.2f}, Distance: {distance:.2f}, Tether Violated: {tether_violated}, Collisions: {collisions}")
        
        steps_taken += 1
        avg_reward = total_reward / (step + 1)  # Calculate average reward based on actual steps taken
        print(f"Episode {episode + 1}: Average Reward: {avg_reward:.2f}")  # Log average reward

        # Log metrics for the episode
        episode_rewards.append(total_reward)
        episode_losses.append(float(loss))
        avg_distance = np.mean(distances) if distances else 0
        reached_goal = env.done

        # Retrieve cumulative reward from the environment's info
        cumulative_reward = env.get_info().get('cumulative_reward', 0)
        print(f"\nEpisode {episode+1} finished with Cumulative Reward: {cumulative_reward}")

        episode_logs.append({
            "episode": episode + 1,
            "reward": total_reward,
            "avg_reward": avg_reward,
            "total_loss": sum(episode_total_loss),
            "total_loss_per_episode": episode_total_loss,
            "policy_loss": sum(episode_policy_loss),
            "policy_loss_per_episode": episode_policy_loss,
            "entropy": sum(episode_entropy_bonus),
            "entropy_per_episode": episode_entropy_bonus,
            "success": reached_goal,
            "tether_violations": tether_violated,
            "collisions": collisions,
            "avg_distance": avg_distance,
            "hyperparams": hyperparams,
            "cumulative_reward": cumulative_reward,
            "out_of_tether_count": info['out_of_tether_count'],
            "steps_taken": steps_taken,
            "communication_count": communication_count,
            "gradient_change": np.sqrt(sum([tf.norm(g).numpy()**2 for g in grads if g is not None])) if grads is not None else 0,
            "episode_accuracies": episode_accuracies,
            "episode_losses": episode_losses,
        })

        if not episode_reset:
            print(f"\nEpisode {episode+1} finished with Reward: {total_reward}")
            print(f"Leader Path: {leader_path}")
            print(f"Follower Path: {follower_path}\n")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Export logs to a CSV file after training
    logs_df = pd.DataFrame(episode_logs)
    log_basepath = os.path.join(os.path.dirname(__file__), f'{"training" if isTraining == True else "tests"}/logs')
    if not os.path.exists(log_basepath):
        os.mkdir(log_basepath)
    FILENAME = os.path.join(log_basepath, "evaluation_metrics.csv")

    # Add timestamp and number of episodes to the logs
    logs_df['timestamp'] = timestamp
    logs_df['num_episodes'] = episodes

    logs_df['algorithm'] = algorithm

    # Append to the file if it exists, otherwise create a new one
    if os.path.exists(FILENAME):
        logs_df.to_csv(FILENAME, mode='a', header=False, index=False)
    else:
        logs_df.to_csv(FILENAME, index=False)

    logs_df.to_csv(FILENAME, index=False)
    print(f"Training logs exported to '{FILENAME}'")


if __name__ == "main":
  env = SimpleGridEnv(
    render_mode="rgb_array", # numpy array representation
    rowSize=10,
    colSize=10,
    num_soft_obstacles=10,
    num_hard_obstacles=5,
    num_robots=2,
    tetherDist=2,
    num_leaders=1,
    num_target=1
  )

  agents_init = [
    Agent(env, role="leader"),
    Agent(env, role="follower")
  ]
  agents = [{"id": agent._id_counter, "role": agent.role} for agent in agents_init]

  leader_pos= np.argwhere(env == LEADER)
  follower_pos= np.argwhere(env == FOLLOWER)
  target_pos = np.argwhere(env == TARGET)


  print(leader_pos)
  print(follower_pos)
  print(target_pos)

  train_MAPPO(2, leader_policy, follower_policy, leader_pos, follower_pos, critic_model, {"lr": 0.001})

  x,y = leader_pos[0]
  env[x,y]

