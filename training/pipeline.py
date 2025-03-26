import os
import sys
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_PATH)
SIMPLEGRID_PATH = os.path.abspath(os.path.join(ROOT_PATH, 'gym-simplegrid', 'gym_simplegrid', 'envs'))
sys.path.append(SIMPLEGRID_PATH)
from simple_grid import SimpleGridEnv
from marl_3 import build_encoder_decoder, build_policy_network, train_MAPPO
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

def main():
    # main pipeline goes here

    # 1. Initialize the Environment
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

    # 2. build encoder-decoder (LSTM)
    encoder_decoder: tf.keras.Model = build_encoder_decoder()

    # 3. build policy network (MAPPO)
    leader_policy_network: tf.keras.Model = build_policy_network() 
    follower_policy_network: tf.keras.Model = build_policy_network()

    # 4. Compile the models
    encoder_decoder.compile(optimizer=Adam, loss='mse')
    leader_policy_network.compile(optimizer=Adam, loss='mse')
    follower_policy_network.compile(optimizer=Adam, loss='mse')

    # 5. Train the models
    train_MAPPO(
        episodes=10,
        leader_model=leader_policy_network,
        follower_model=follower_policy_network,
        encoder_decoder=encoder_decoder,
        env=env,
        lr=0.001
    )

    # 6. Save the models
    if not (os.path.exists('models')):
        os.mkdir('models')
    encoder_decoder.save('models/encoder_decoder.h5')
    leader_policy_network.save('models/leader_policy_network.h5')
    follower_policy_network.save('models/follower_policy_network.h5')
    print(f"Models are saved.")

if __name__ == "main":
    main()