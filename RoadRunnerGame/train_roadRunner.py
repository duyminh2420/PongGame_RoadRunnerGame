# UNC Charlotte
# ITCS 5153 - Applied AI - Fall 2023
# Lab 5
# Reinforcement Learning
# This module implements Reinforcement Learning solutions in Python, using AI Gym libraries and RL libraries to train and test a model. 
# Student ID: 801131341
####### Answer the questions:
# Name env: RoadRunnerNoFrameskip-v0
# Type algorithm: Deepq
# Configuration parameters: convs; hiddens; dueling; total_timesteps; buffer_size, exploration_fraction; exploration_fiinal_eps; train_freq; learning_starts; target_network_update_freq; gamma
# Time train the model: 2hrs 30 mins
# No GPU was used
# My GPU: UHD grpahic and Nvidia Geforce GTX 3060
# OS version: Window 11 
# My CPU: 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz   2.30 GHz

from baselines import deepq
from baselines import bench
from baselines import logger
from baselines.common.atari_wrappers import make_atari


def main():
    logger.configure()
    env = make_atari('RoadRunnerNoFrameskip-v0')
    env = bench.Monitor(env, logger.get_dir())
    env = deepq.wrap_atari_dqn(env)

    model = deepq.learn(
        env,
        "conv_only",
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[256],
        dueling=True,
        lr=1e-4,
        #total_timesteps=int(2e7),
        total_timesteps=int(1e5),
        buffer_size=10000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=10000,
        target_network_update_freq=1000,
        gamma=0.99,
    )

    model.save('RoadRunner_model_801131341.pkl')
    env.close()

if __name__ == '__main__':
    main()
