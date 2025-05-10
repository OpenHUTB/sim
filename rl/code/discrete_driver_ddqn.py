import sys
# 添加Carla Python API路径
sys.path.append(r'A:\carla\WindowsNoEditor_9.15\PythonAPI\carla\dist\carla-0.9.15-py3.7-win-amd64.egg')

import os
import sys
import time
import random
import numpy as np
import argparse
import logging
import pickle
import torch
from distutils.util import strtobool
from threading import Thread
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from simulation.connection import ClientConnection
from simulation.environment import CarlaEnvironment
from networks.off_policy.ddqn.agent import DQNAgent
from encoder_init import EncodeState
from parameters import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default='ddqn', help='name of the experiment')
    parser.add_argument('--env-name', type=str, default='carla', help='name of the simulation environment')
    parser.add_argument('--learning-rate', type=float, default=DQN_LEARNING_RATE, help='learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=SEED, help='seed of the experiment')
    parser.add_argument('--total-episodes', type=int, default=EPISODES, help='total timesteps of the experiment')
    parser.add_argument('--train', type=bool, default=False, help='is it training?')
    parser.add_argument('--town', type=str, default="Town10HD", help='which town do you like?')
    parser.add_argument('--load-checkpoint', type=bool, default=MODEL_LOAD, help='resume training?')
    parser.add_argument('--num-actions', type=int, default=NUM_ACTIONS, help='num of discrete actions')
    parser.add_argument('--torch-deterministic', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, cuda will not be enabled by deafult')
    args = parser.parse_args()

    return args


def runner():
    # ========================================================================
    #                           BASIC PARAMETER & LOGGING SETUP
    # ========================================================================

    args = parse_args()
    exp_name = args.exp_name
    town = args.town
    train = args.train
    checkpoint_load = args.load_checkpoint
    num_actions = args.num_actions

    try:
        if exp_name == 'ddqn':
            run_name = f"DDQN"
    except Exception as e:
        print(e.message)
        sys.exit()

    if train == True:
        writer = SummaryWriter(f"runs/{run_name}/{town}")
    else:
        writer = SummaryWriter(f"runs/{run_name}_TEST/{town}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}" for key, value in vars(args).items()])))

    # Seeding to reproduce the results
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    n_actions = num_actions  # Car can only make 7 actions for steer
    epoch = 0
    episode = 0
    timestep = 0
    cumulative_score = 0
    episodic_length = list()
    scores = list()
    deviation_from_center = 0
    distance_covered = 0

    # ========================================================================
    #                           CREATING THE SIMULATION
    # ========================================================================
    try:
        client, world = ClientConnection(town).setup()
        logging.info("Connection has been setup successfully.")
    except:
        logging.error("Connection has been refused by the server.")
        ConnectionRefusedError
    if train:
        env = CarlaEnvironment(client, world, town, continuous_action=False, algorithm='dqn', route_mode='1')
    else:
        env = CarlaEnvironment(client, world, town, checkpoint_frequency=None, continuous_action=False, algorithm='dqn', route_mode='1')
    encode = EncodeState(LATENT_DIM)


    time.sleep(0.5)
    # ========================================================================
    #                           ALGORITHM
    # ========================================================================
    if train is False:  # Test
        agent = DQNAgent(town, n_actions)
        agent.load_model()
        for params in agent.q_network_eval.parameters():
            params.requires_grad = False
        for params in agent.q_network_target.parameters():
            params.requires_grad = False
    else:  # Training
        # resume
        if checkpoint_load:
            agent = DQNAgent(town, n_actions)
            agent.load_model()
            if exp_name == 'ddqn':
                with open(f'checkpoints/DDQN/{town}/checkpoint_ddqn.pickle', 'rb') as f:
                    data = pickle.load(f)
                    epoch = data['epoch']
                    cumulative_score = data['cumulative_score']
                    agent.epsilon = data['epsilon']
        else:
            agent = DQNAgent(town, n_actions)

    if exp_name == 'ddqn' and checkpoint_load:
        #
        while agent.replay_buffer.counter < agent.replay_buffer.buffer_size:
            observation = env.reset()
            observation = encode.process(observation)
            done = False
            while not done:
                action = random.randint(0, n_actions - 1)
                new_observation, reward, done, _ = env.step(action)
                new_observation = encode.process(new_observation)
                agent.save_transition(observation, action, reward, new_observation, int(done))
                observation = new_observation

    if args.train:
        print("-----------------Carla_DQN Train-------------------")
        for step in range(epoch + 1, EPISODES + 1):
            # Reset
            done = False
            observation = env.reset()
            observation = encode.process(observation)
            current_ep_reward = 0

            # Episode start: timestamp
            t1 = datetime.now()

            while not done:
                action = agent.get_action(args.train, observation)
                new_observation, reward, done, info = env.step(action)

                if new_observation is None:
                    break
                new_observation = encode.process(new_observation)
                current_ep_reward += reward

                agent.save_transition(observation, action, reward, new_observation, int(done))
                if agent.get_len_buffer() > WARMING_UP:  # begin train if buffer size > WARMING_UP
                    agent.learn()  # DQN is learned in every step

                observation = new_observation
                timestep += 1
                if done:
                    episode += 1

            # Episode end : timestamp
            t2 = datetime.now()
            t3 = t2 - t1
            episodic_length.append(abs(t3.total_seconds()))

            deviation_from_center += info[1]
            distance_covered += info[0]

            scores.append(current_ep_reward)

            if checkpoint_load:
                cumulative_score = ((cumulative_score * (episode - 1)) + current_ep_reward) / (episode)
            else:
                cumulative_score = np.mean(scores)

            print('Starting Episode: ', episode, ', Epsilon Now:  {:.3f}'.format(agent.epsilon),
                  'Reward:  {:.2f}'.format(current_ep_reward), ', Average Reward:  {:.2f}'.format(cumulative_score))
            agent.save_model(current_ep_reward, step)  # only save the best

            if episode % 100 == 0:
                data_obj = {'cumulative_score': cumulative_score, 'epsilon': agent.epsilon, 'epoch': step}
                with open(f'checkpoints/DDQN/{town}/checkpoint_ddqn.pickle', 'wb') as handle:
                    pickle.dump(data_obj, handle)

            if episode % 10 == 0:
                writer.add_scalar("Cumulative Reward/info", cumulative_score, episode)
                writer.add_scalar("Epsilon/info", agent.epsilon, episode)
                writer.add_scalar("Episodic Reward/episode", scores[-1], episode)
                writer.add_scalar("Reward/(t)", current_ep_reward, timestep)
                writer.add_scalar("Episode Length (s)/info", np.mean(episodic_length), episode)
                writer.add_scalar("Average Deviation from Center/episode", deviation_from_center / 10, episode)
                writer.add_scalar("Average Distance Covered (m)/episode", distance_covered / 10, episode)

                episodic_length = list()
                deviation_from_center = 0
                distance_covered = 0

        print("Terminating the run.")
        sys.exit()
    else:
        # Testing
        for step in range(epoch + 1, EPISODES + 1):
            # Reset
            done = False
            observation = env.reset()
            observation = encode.process(observation)
            current_ep_reward = 0

            # Episode start: timestamp
            t1 = datetime.now()

            while not done:
                action = agent.get_action(args.train, observation)
                new_observation, reward, done, info = env.step(action)

                if new_observation is None:
                    break
                new_observation = encode.process(new_observation)
                current_ep_reward += reward
                observation = new_observation

            # Episode end : timestamp
            t2 = datetime.now()
            t3 = t2 - t1
            episodic_length.append(abs(t3.total_seconds()))

            deviation_from_center += info[1]
            distance_covered += info[0]

            scores.append(current_ep_reward)

            if checkpoint_load:
                cumulative_score = ((cumulative_score * (step - 1)) + current_ep_reward) / (step)
            else:
                cumulative_score = np.mean(scores)

            print('Starting Episode: ', step, ', Epsilon Now:  {:.3f}'.format(agent.epsilon),
                  'Reward:  {:.2f}'.format(current_ep_reward), ', Average Reward:  {:.2f}'.format(cumulative_score))

            writer.add_scalar("TEST: Episodic Reward/episode", scores[-1], step)
            writer.add_scalar("TEST: Cumulative Reward/info", cumulative_score, step)
            writer.add_scalar("TEST: Episode Length (s)/info", np.mean(episodic_length), step)
            writer.add_scalar("TEST: Deviation from Center/episode", deviation_from_center, step)
            writer.add_scalar("TEST: Distance Covered (m)/episode", distance_covered, step)

            episodic_length = list()
            deviation_from_center = 0
            distance_covered = 0

        print("Terminating the run.")
        sys.exit()


if __name__ == "__main__":
    runner()
    # try:
    #     runner()
    #
    # except KeyboardInterrupt:
    #     sys.exit()
    # finally:
    #     print('\nExit')
