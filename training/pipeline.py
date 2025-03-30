import os
import sys
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_PATH)
SIMPLEGRID_PATH = os.path.abspath(os.path.join(ROOT_PATH, 'gym-simplegrid', 'gym_simplegrid', 'envs'))
sys.path.append(SIMPLEGRID_PATH)
from simple_grid import SimpleGridEnv
from marl_3 import train_MAPPO, encoder_decoder, leader_policy, follower_policy
import time
import tensorflow as tf

# Hyperparemeter Tuning
import random

def main(): # main pipeline goes here
    # Remove the evaluation CSV if it exists
    CSVPATH = 'logs/evaluation.csv'
    if os.path.exists(CSVPATH):
        os.remove(CSVPATH)
    
    # Tracking GPU
    tf.debugging.set_log_device_placement(True)

    tf.profiler.experimental.start('logs') # start GPU memory count
    startTime = time.time()

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
    HYPERPARAMETER_COUNT = 2
    learning_rates = [round(random.uniform(0.0001, 0.01), 6) for _ in range(HYPERPARAMETER_COUNT)]
    episodes_list = [50] # Now we only consider a static number of episodes for simplicity
    contrastive_weights = [round(random.uniform(0.1, 1.0), 2) for _ in range(HYPERPARAMETER_COUNT)]
    reconstruction_weights = [round(random.uniform(0.1, 0.5), 2) for _ in range(HYPERPARAMETER_COUNT)]
    entropy_weights = [round(random.uniform(0.01, 0.1), 3) for _ in range(HYPERPARAMETER_COUNT)]
    max_steps_list = [100]

    best_score = -float('inf')
    best_params = None

    algoStartTime = time.time()

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
                                encoded_model=encoder_decoder,  # Corrected argument name
                                env=env,
                                hyperparams=params
                            )

                            # Evaluate the models (e.g., based on cumulative reward or success rate)
                            # Retrieve cumulative reward from the environment's info
                            success_rate = env.get_info().get('cumulative_reward', 0)

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
    print(f"Time taken for hyperparameter tuning and training with the best ones: {endTime - algoStartTime} seconds")
    print(f"Time taken for the entire pipeline process is: {endTime - startTime} seconds")

    tf.profiler.experimental.stop() # end GPU memory count

    # Print Best Hyperparameters
    print(f"Best Hyperparameters: {best_params}") # dict: best learning rate + episode
    print(f"Best Success Rate: {best_score}") # best success rate

    print(f"✅ Completed the training pipeline successfully.")

if __name__ == "__main__":
    main()