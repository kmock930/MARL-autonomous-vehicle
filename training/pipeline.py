import os
import sys
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_PATH)
SIMPLEGRID_PATH = os.path.abspath(os.path.join(ROOT_PATH, 'gym-simplegrid', 'gym_simplegrid', 'envs'))
sys.path.append(SIMPLEGRID_PATH)
from simple_grid import SimpleGridEnv
from marl_3 import train_MAPPO, encoder_decoder, leader_policy, follower_policy, MAPPO
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import time

# Hyperparemeter Tuning
import random

def main(): # main pipeline goes here

    # Initialize the Environment
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

    if not (os.path.exists('models')):
        os.mkdir('models')

    # Load and Save the pretrained RAW models
    leader_policy.save('models/leader_model_RAW.h5')
    follower_policy.save('models/follower_model_RAW.h5')
    encoder_decoder.save('models/policy_network_RAW.h5')

    # Define Hyperparameter Grid
    learning_rates = [round(random.uniform(0.0001, 0.01), 6) for _ in range(5)]
    episodes_list = [50] # Now we only consider a static number of episodes for simplicity
    contrastive_weights = [round(random.uniform(0.1, 1.0), 2) for _ in range(5)]
    reconstruction_weights = [round(random.uniform(0.1, 0.5), 2) for _ in range(5)]
    entropy_weights = [round(random.uniform(0.01, 0.1), 3) for _ in range(5)]
    max_steps_list = [random.choice([50, 100, 200]) for _ in range(5)]

    best_score = -float('inf')
    best_params = None

    startTime = time.time()

    # Perform Hyperparameter Tuning with Grid Search
    for lr in learning_rates:
        for episodes in episodes_list:
            startTimeInEpisode = time.time()

            for contrastive_weight in contrastive_weights:
                for reconstruction_weight in reconstruction_weights:
                    for entropy_weight in entropy_weights:
                        for max_steps in max_steps_list:
                            # Construct the Grid
                            params = {
                                "lr": lr, 
                                "episodes": episodes,
                                "contrastive_weight": contrastive_weight,
                                "reconstruction_weight": reconstruction_weight,
                                "entropy_weight": entropy_weight,
                                "max_steps": max_steps
                            }
                            print(f"Training with parameters: {params}")

                            # Train the models
                            train_MAPPO(
                                episodes=episodes,
                                leader_model=leader_policy,
                                follower_model=follower_policy,
                                encoder_decoder=encoder_decoder,
                                env=env,
                                hyperparams=params
                            )

                            # Evaluate the models (e.g., based on cumulative reward or success rate)
                            # For simplicity, assume train_MAPPO returns a success rate
                            success_rate = env.cumulative_reward  # Replace with actual metric if available

                            print(f"Success Rate: {success_rate}")

                            # Save the best model
                            if success_rate > best_score:
                                best_score = success_rate
                                best_params = {"lr": lr, "episodes": episodes}

                                # Save the best models
                                leader_policy.save('models/best_leader_model.h5')
                                follower_policy.save('models/best_follower_model.h5')
                                encoder_decoder.save('models/best_policy_network.h5')
                                print(f"Best models are saved.")

            endTimeInEpisode = time.time()
            print(f"Time taken for this episode: {endTimeInEpisode - startTimeInEpisode} seconds")

    endTime = time.time()
    print(f"Time taken for hyperparameter tuning and training with the best ones: {endTime - startTime} seconds")

    # Print Best Hyperparameters
    print(f"Best Hyperparameters: {best_params}") # dict: best learning rate + episode
    print(f"Best Success Rate: {best_score}") # best success rate

if __name__ == "main":
    main()