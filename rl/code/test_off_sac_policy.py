# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import argparse
import os
import gym
import time
import json
from agent.deepmdp_sac import DeepMDPAgent_SAC
from agent.deepmdp_ddpg import DeepMDPAgent_DDPG
from agent.deepmdp_td3 import DeepMDPAgent_TD3
from agent.deepmdp_ddqn import DeepMDPAgent_DDQN
from utils import utils
from utils.logger import Logger
from utils.video import VideoRecorder
from carla_env.carla_env import CarlaEnv
import sys


def make_agent(obs_shape, action_shape, args, device):
    # take deepmdp as base
    print("select agent:", args.agent)
    if args.agent == 'sac':
        agent = DeepMDPAgent_SAC(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            encoder_stride=args.encoder_stride,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            decoder_type=args.decoder_type,
            decoder_lr=args.decoder_lr,
            decoder_update_freq=args.decoder_update_freq,
            decoder_weight_lambda=args.decoder_weight_lambda,
            transition_model_type=args.transition_model_type,
            num_layers=args.num_layers,
            num_filters=args.num_filters
        )
    elif args.agent == 'ddpg':
        agent = DeepMDPAgent_DDPG(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            encoder_stride=args.encoder_stride,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            decoder_type=args.decoder_type,
            decoder_lr=args.decoder_lr,
            decoder_update_freq=args.decoder_update_freq,
            decoder_weight_lambda=args.decoder_weight_lambda,
            transition_model_type=args.transition_model_type,
            num_layers=args.num_layers,
            num_filters=args.num_filters
        )
    elif args.agent == 'td3':
        agent = DeepMDPAgent_TD3(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            encoder_stride=args.encoder_stride,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            decoder_type=args.decoder_type,
            decoder_lr=args.decoder_lr,
            decoder_update_freq=args.decoder_update_freq,
            decoder_weight_lambda=args.decoder_weight_lambda,
            transition_model_type=args.transition_model_type,
            num_layers=args.num_layers,
            num_filters=args.num_filters
        )
    elif args.agent == 'ddqn':
        agent = DeepMDPAgent_DDQN(
            obs_shape=obs_shape,
            action_shape=[7],  # 离散，比较特殊，根据action_space的数量定
            device=device,
            hidden_dim=args.hidden_dim,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            encoder_stride=args.encoder_stride,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            decoder_type=args.decoder_type,
            decoder_lr=args.decoder_lr,
            decoder_update_freq=args.decoder_update_freq,
            decoder_weight_lambda=args.decoder_weight_lambda,
            transition_model_type=args.transition_model_type,
            num_layers=args.num_layers,
            num_filters=args.num_filters
        )

    else:
        print("Select the correct algorithm!!!")
        sys.exit()

    return agent


def evaluate(env, agent, video, num_episodes, args, do_carla_metrics=False, res_dir=""):
    # carla metrics:
    reason_each_episode_ended = []
    distance_driven_each_episode = []
    crash_intensity = 0.
    steer = 0.
    brake = 0.
    count = 0

    for i in range(num_episodes):
        # carla metrics:
        dist_driven_this_episode = 0.

        obs = env.reset()
        video.init(enabled=True)
        done = False
        episode_reward = 0
        while not done:
            with utils.eval_mode(agent):
                # obs: (9, 84, 84 * num_cameras)
                if args.agent == "ddqn":  # 只有这玩意是离散的
                    action_idx = agent.select_action(obs)
                    # 记得跟自己实验的动作空间保持一致
                    action_space = np.array([
                        -0.50,
                        -0.30,
                        -0.10,
                        0.0,
                        0.10,
                        0.30,
                        0.50
                    ])
                    steer = action_space[action_idx]  # ddqn用来学习转向
                    throttle = 0.5  # 速度固定
                    action = np.array([steer, throttle])
                else:
                    action = agent.select_action(obs)

            obs, reward, done, info = env.step(action)
            if do_carla_metrics:
                if info is not None:
                    dist_driven_this_episode += info.get('distance', 0)
                    crash_intensity += info.get('crash_intensity', 0)
                    steer += abs(info.get('steer', 0))
                    brake += info.get('brake', 0)
                    count += 1
                else:
                    print("Warning: 'info' is None, skipping metrics update.")

            video.record(env)
            episode_reward += reward

        if do_carla_metrics:
            distance_driven_each_episode.append(dist_driven_this_episode)
            eval_log_filepath = os.path.join(res_dir, "eval_log.txt")
            eval_log_txt_formatter = "{step},{distance_driven_each_episode},{episode_reward}\n"
            to_write = eval_log_txt_formatter.format(step=i,
                                                     distance_driven_each_episode=dist_driven_this_episode,
                                                     episode_reward=episode_reward)

            with open(eval_log_filepath, "a") as f:
                f.write(to_write)

        video.save('%d.mp4' % i)


    if do_carla_metrics:
        print('METRICS--------------------------')
        print("distance_driven_each_episode: {}".format(distance_driven_each_episode))
        print('crash_intensity: {}'.format(crash_intensity / num_episodes))
        print('steer: {}'.format(steer / count))
        print('brake: {}'.format(brake / count))
        print('---------------------------------')


# 参数要跟训练的保持一致
def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--task_name', default='carla', choices=['carla'])
    parser.add_argument('--scenarios', default='highway', choices=['highway', 'ghost_static'])
    parser.add_argument('--image_size', default=84, type=int)
    parser.add_argument('--action_repeat', default=4, type=int)
    parser.add_argument('--frame_stack', default=3, type=int)
    parser.add_argument('--agent', default='ddpg', type=str, choices=['sac', 'ddpg', 'td3', 'ddqn'])
    parser.add_argument('--init_steps', default=1000, type=int)
    parser.add_argument('--num_train_steps', default=1000000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--k', default=3, type=int, help='number of steps for inverse model')
    parser.add_argument('--bisim_coef', default=0.5, type=float, help='coefficient for bisim terms')
    # critic
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--critic_beta', default=0.9, type=float)
    parser.add_argument('--critic_tau', default=0.005, type=float)
    parser.add_argument('--critic_target_update_freq', default=2, type=int)
    # actor
    parser.add_argument('--actor_lr', default=1e-3, type=float)
    parser.add_argument('--actor_beta', default=0.9, type=float)
    parser.add_argument('--actor_log_std_min', default=-10, type=float)
    parser.add_argument('--actor_log_std_max', default=2, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)
    # encoder/decoder
    parser.add_argument('--encoder_type', default='pixelCarla', type=str, choices=['pixel', 'pixelCarla', 'identity'])
    parser.add_argument('--encoder_feature_dim', default=50, type=int)
    parser.add_argument('--encoder_lr', default=1e-3, type=float)
    parser.add_argument('--encoder_tau', default=0.005, type=float)
    parser.add_argument('--encoder_stride', default=1, type=int)
    parser.add_argument('--decoder_type', default='identity', type=str, choices=['pixel', 'identity'])
    parser.add_argument('--decoder_lr', default=1e-3, type=float)
    parser.add_argument('--decoder_update_freq', default=1, type=int)
    parser.add_argument('--decoder_weight_lambda', default=0.0, type=float)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--num_filters', default=32, type=int)
    # sac
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.01, type=float)
    parser.add_argument('--alpha_lr', default=1e-3, type=float)
    parser.add_argument('--alpha_beta', default=0.9, type=float)
    parser.add_argument('--latent_dim', default=128, type=int)
    # misc
    parser.add_argument('--seed', default=3, type=int)
    parser.add_argument('--work_dir', default='./log', type=str)
    parser.add_argument('--save_tb', default=True, action='store_true')
    parser.add_argument('--save_video', default=True, action='store_true')
    parser.add_argument('--render', default=False, action='store_true')
    parser.add_argument('--port', default=2000, type=int)
    parser.add_argument('--trafficManagerPort', default=2050, type=int)
    parser.add_argument('--detach_encoder', default=False, action='store_true')
    parser.add_argument('--transition_model_type', default='deterministic', type=str, choices=['', 'deterministic', 'probabilistic', 'ensemble'])
    parser.add_argument('--critic_best_path', default="C:/doctor/Carla_Deepmdp_RL/log/highway_ddpg_seed_1_2025-04-17-12-54-27/model", type=str)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    utils.set_seed_everywhere(args.seed)

    env = CarlaEnv(
        render_display=args.render,  # for local debugging only
        display_text=args.render,  # for local debugging only
        changing_weather_speed=0.1,  # [0, +inf)
        rl_image_size=args.image_size,
        max_episode_steps=1000,
        frame_skip=args.action_repeat,
        port=args.port,
        trafficManagerPort=args.trafficManagerPort,
        scenarios=args.scenarios
    )

    # stack several consecutive frames together
    if args.encoder_type.startswith('pixel'):
        env = utils.FrameStack(env, k=args.frame_stack)

    # add time, windows目前可能无效,可以改成有效的唯一方式
    work_dir = args.work_dir + '/Eval_' + args.agent + '_' + args.scenarios + '_' + "seed_{}_".format(args.seed) + time.strftime(
        "%Y-%m-%d-%H-%M-%S")
    utils.make_dir(work_dir)
    video_dir = utils.make_dir(os.path.join(work_dir, 'video'))
    res_dir = utils.make_dir(os.path.join(work_dir, 'res_dir'))

    video = VideoRecorder(video_dir if args.save_video else None)

    with open(os.path.join(work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # the dmc2gym wrapper standardizes actions
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    L = Logger(work_dir, use_tb=args.save_tb)

    agent = make_agent(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        args=args,
        device=device
    )
    print(args.critic_best_path)
    agent.load_best(args.critic_best_path)
    # agent.load(args.critic_best_path, 99999)
    # 测几次，可以多测很多次，挑表现最好的
    num_eval_episodes = 10
    evaluate(env, agent, video, num_eval_episodes, args, do_carla_metrics=True, res_dir=res_dir)


if __name__ == '__main__':
    main()
