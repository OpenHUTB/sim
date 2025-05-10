# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import sys

import numpy as np
import torch
import argparse
import os
import time
import json

from utils import utils
from utils.logger import Logger
from utils.video import VideoRecorder

from agent.deepmdp_sac import DeepMDPAgent_SAC
from agent.deepmdp_ddpg import DeepMDPAgent_DDPG
from agent.deepmdp_td3 import DeepMDPAgent_TD3
from agent.deepmdp_ddqn import DeepMDPAgent_DDQN

from carla_env.carla_env import CarlaEnv


def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--domain_name', default='carla', choices=['carla'])
    parser.add_argument('--scenarios', default='city01', choices=['highway', 'ghost_static','city01'])
    parser.add_argument('--image_size', default=84, type=int)
    parser.add_argument('--action_repeat', default=3, type=int)
    parser.add_argument('--frame_stack', default=3, type=int)
    parser.add_argument('--img_source', default=None, type=str, choices=['color', 'noise', 'images', 'video', 'none'])
    # train
    parser.add_argument('--agent', default='sac', type=str, choices=['sac', 'ddpg', 'td3', 'ddqn'])
    parser.add_argument('--init_steps', default=1000, type=int)  #
    """
    replay buffer，不同算法可能不一样，10000 or 1000,根据跑起来的经验, 
    sac, ddpg, td3: 10000
    ddqn: 10000
    """
    parser.add_argument('--replay_buffer_capacity', default=10000, type=int)
    """
    好像如果有路线导航，学习收敛都比较快(不同场景不同算法可能不一样)，可以设置为5w这样子或者更低，省点训练时间，如果去掉导航只看行驶距离，可以设置为10w。
    """
    parser.add_argument('--num_train_steps', default=30000, type=int)  # 反正自己灵活调整
    """
    batch_size可以调整,
    sac, ddpg: 128
    td3: 256
    ddqn: 64
    """
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--k', default=3, type=int, help='number of steps for inverse model')
    parser.add_argument('--bisim_coef', default=0.5, type=float, help='coefficient for bisim terms')
    parser.add_argument('--load_encoder', default=False, type=bool)
    parser.add_argument('--encoder_path', default="", type=str)
    parser.add_argument('--model_path', default="", type=str)

    # eval
    parser.add_argument('--eval_freq', default=1000, type=int)  # TODO: master had 10000
    parser.add_argument('--num_eval_episodes', default=5, type=int)
    # critic
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--critic_beta', default=0.9, type=float)
    parser.add_argument('--critic_tau', default=0.01, type=float)
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
    parser.add_argument('--encoder_tau', default=0.05, type=float)
    parser.add_argument('--encoder_stride', default=1, type=int)
    parser.add_argument('--decoder_type', default='identity', type=str, choices=['pixel', 'identity'])
    parser.add_argument('--decoder_lr', default=1e-3, type=float)
    parser.add_argument('--decoder_update_freq', default=1, type=int)
    parser.add_argument('--decoder_weight_lambda', default=0.0000001, type=float)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--num_filters', default=32, type=int)
    #
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=1e-4, type=float)
    parser.add_argument('--alpha_beta', default=0.5, type=float)
    # misc
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--work_dir', default='./log', type=str)
    parser.add_argument('--save_tb', default=True, action='store_true')
    parser.add_argument('--save_model', default=True, action='store_true')
    parser.add_argument('--save_buffer', default=False, action='store_true')
    parser.add_argument('--save_video', default=True, action='store_true')
    parser.add_argument('--render', default=False, action='store_true')
    parser.add_argument('--transition_model_type', default='', type=str,
                        choices=['', 'deterministic', 'probabilistic', 'ensemble'])
    parser.add_argument('--port', default=2000, type=int)
    parser.add_argument('--trafficManagerPort', default=2050, type=int)
    args = parser.parse_args()
    return args


def evaluate(env, agent, video, num_episodes, args, do_carla_metrics=False, res_dir="", model_dir=""):
    # carla metrics:
    reason_each_episode_ended = []
    distance_driven_each_episode = []
    crash_intensity = 0.
    steer = 0.
    brake = 0.
    count = 0
    agent.load_best(model_dir)  #
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
                    #
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
        sys.exit()


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

    if args.load_encoder:
        # model_dict = agent.actor.encoder.state_dict()
        # encoder_dict = torch.load(args.encoder_path)
        # encoder_dict = {k[8:]: v for k, v in encoder_dict.items() if 'encoder.' in k}  # hack to remove encoder. string
        # agent.actor.encoder.load_state_dict(encoder_dict)
        # agent.critic.encoder.load_state_dict(encoder_dict)
        agent.load_best(args.model_path)

    return agent


def main():
    args = parse_args()
    utils.set_seed_everywhere(args.seed)

    env = CarlaEnv(
        render_display=args.render,  # for local debugging only
        display_text=args.render,  # for local debugging only
        changing_weather_speed=0.1,  # [0, +inf)
        rl_image_size=args.image_size,
        max_episode_steps=2000,  # 这个最大帧数对于定点导航任务来说其实可以去掉，或者调大， 不是导航任务一般是10000
        frame_skip=args.action_repeat,
        port=args.port,
        trafficManagerPort=args.trafficManagerPort,
        scenarios=args.scenarios,
        algorithm=args.agent
    )
    # TODO: implement env.seed(args.seed) ?
    eval_env = env

    # stack several consecutive frames together
    if args.encoder_type.startswith('pixel'):
        env = utils.FrameStack(env, k=args.frame_stack)
        eval_env = utils.FrameStack(eval_env, k=args.frame_stack)

    # add time
    work_dir = args.work_dir + '/' + args.scenarios + "_" + args.agent + "_seed_{}_".format(args.seed) + time.strftime(
        "%Y-%m-%d-%H-%M-%S")
    utils.make_dir(work_dir)
    video_dir = utils.make_dir(os.path.join(work_dir, 'video'))
    model_dir = utils.make_dir(os.path.join(work_dir, 'model'))
    buffer_dir = utils.make_dir(os.path.join(work_dir, 'buffer'))
    res_dir = utils.make_dir(os.path.join(work_dir, 'res_dir'))

    video = VideoRecorder(video_dir if args.save_video else None)

    with open(os.path.join(work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # the dmc2gym wrapper standardizes actions
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    if args.agent == "ddqn":  # 只有这玩意是离散的
        replay_buffer = utils.ReplayBuffer_DDQN(
            obs_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            capacity=args.replay_buffer_capacity,
            batch_size=args.batch_size,
            device=device
        )
    else:
        replay_buffer = utils.ReplayBuffer(
            obs_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            capacity=args.replay_buffer_capacity,
            batch_size=args.batch_size,
            device=device
        )

    agent = make_agent(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        args=args,
        device=device
    )

    L = Logger(work_dir, use_tb=args.save_tb)

    # train and evaluate
    episode, episode_reward, done = 0, 0, True
    expl_noise = 0.1
    start_time = time.time()
    for step in range(args.num_train_steps):
        if done:
            if args.decoder_type == 'inverse':
                for i in range(1, args.k):  # fill k_obs with 0s if episode is done
                    replay_buffer.k_obses[replay_buffer.idx - i] = 0
            if step > 0:
                L.log('train/duration', time.time() - start_time, step)
                start_time = time.time()
                L.dump(step)

            if args.save_model:
                # agent.save(model_dir, step)
                # print('----------------save model-----------------')
                agent.save_best(model_dir, episode_reward)

            # evaluate agent periodically
            # if episode % args.eval_freq == 0 and episode != 0:
            #     print('----------------start eval-----------------')
            #     L.log('eval/episode', episode, step)
            #     evaluate(env, agent, video, args.num_eval_episodes, args,
            #              do_carla_metrics=True, res_dir=res_dir, model_dir=model_dir)
            #     # if args.save_buffer:
            #     #     replay_buffer.save(buffer_dir)

            L.log('train/episode_reward', episode_reward, step)

            train_log_filepath = os.path.join(res_dir, "train_log.txt")
            train_log_txt_formatter = "{step}:{episode_reward}\n"
            to_write = train_log_txt_formatter.format(step=step,
                                                      episode_reward=episode_reward)
            with open(train_log_filepath, "a") as f:
                f.write(to_write)

            obs = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1
            reward = 0

            L.log('train/episode', episode, step)

        # 不同算法的交互
        if args.agent == "ddqn":  # 只有这玩意是离散的
            action = agent.sample_action(obs)
            action_space = np.array([
                -0.50,
                -0.30,
                -0.10,
                0.0,
                0.10,
                0.30,
                0.50
            ])
            steer = action_space[action]  # ddqn用来学习转向
            throttle = 0.5  # 速度固定
            action_env = np.array([steer, throttle])

        elif args.agent == "td3":
            # sample action for data collection
            if step < args.init_steps:
                action = env.action_space.sample()  # random
            else:
                # expl_noise *= 0.999  # 不断减少
                noise = np.random.normal(0, expl_noise, size=2)  # 持续加噪声
                action = agent.sample_action(obs) + noise
                action = action.clip(-1, 1)
            action_env = action

        else:  # sac ddpg
            # sample action for data collection
            if step < args.init_steps:
                action = env.action_space.sample()  # random
            else:
                with utils.eval_mode(agent):
                    action = agent.sample_action(obs)
            action_env = action

        # run training update
        if args.agent == "ddqn":
            if step > args.batch_size:
                agent.update(replay_buffer, L, step)

        elif args.agent == "td3":
            if step >= args.init_steps:
                agent.update(replay_buffer, L, step)
        else:
            if step >= args.init_steps:  # args.init_steps相当于warming up的步数
                num_updates = args.init_steps if step == args.init_steps else 1
                for _ in range(num_updates):
                    agent.update(replay_buffer, L, step)

        curr_reward = reward
        next_obs, reward, done, _ = env.step(action_env)
        # allow infinit bootstrap
        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(
            done
        )
        episode_reward += reward

        replay_buffer.add(obs, action, curr_reward, reward, next_obs, done_bool)
        np.copyto(replay_buffer.k_obses[replay_buffer.idx - args.k], next_obs)
        # update current obs
        obs = next_obs
        episode_step += 1

    print('----------------start eval-----------------')
    evaluate(env, agent, video, args.num_eval_episodes, args, do_carla_metrics=True, res_dir=res_dir,
             model_dir=model_dir)


if __name__ == '__main__':
    main()
