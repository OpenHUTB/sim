# experiments/run_attention_dqn.py — Attention-DQN 训练入口（创新算法）
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_algorithms.dqn import AttentionDQNAgent
from utils.data_saver import init_result_dir, save_training_log, save_model, save_metrics
from utils.metrics import MovingAverage
from config.dqn_config import MAX_EPISODES, SAVE_EVERY_N_EPISODES, LOG_EVERY_N_EPISODES
from scenarios import create_scenario

DEFAULT_SCENARIOS = ["heavy_fog", "tunnel_night", "ghost_peek", "night_pedestrian", "fog_ghost"]


def run(scenario_name="heavy_fog", max_episodes=MAX_EPISODES,
        render=False, model_dir="models/attention_dqn"):
    scenario = create_scenario(scenario_name)
    env = scenario.create_env()
    print("场景: %s | 类别: %s" % (scenario.name, scenario.category))

    agent = AttentionDQNAgent(state_size=env.observation_space, action_size=env.action_space)
    print("智能体: Attention-DQN | Heads: %d" % agent.q_net.num_heads)

    dirs = init_result_dir("attn_dqn_%s" % scenario_name, base_dir="results/attention_dqn")
    model_path = os.path.join(model_dir, "attn_dqn_%s.pth" % scenario_name)
    log_path = os.path.join(dirs["logs"], "training_log.csv")
    os.makedirs(model_dir, exist_ok=True)

    reward_ma = MovingAverage(window_size=100)
    best_reward = -float("inf")

    for ep in range(1, max_episodes + 1):
        agent.start_episode()
        state = env.reset()
        scenario.spawn_scenario_actors(env)
        episode_reward, steps = 0.0, 0
        while True:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.store(state, action, reward, next_state, done)
            agent.train(); agent.total_steps += 1
            scenario.step_callback(steps)
            state = next_state; episode_reward += reward; steps += 1
            if render and ep % 50 == 0: env.render()
            if done: break
        agent.end_episode(episode_reward)
        reward_ma.update(episode_reward); avg = reward_ma.mean()
        if ep % LOG_EVERY_N_EPISODES == 0:
            loss_val = agent.last_loss or 0.0
            print("Ep %4d | Reward: %7.1f | Avg100: %7.1f | Eps: %.3f | Loss: %.5f"
                  % (ep, episode_reward, avg, agent.epsilon, loss_val))
            save_training_log(log_path, {"episode": ep, "reward": round(episode_reward, 2),
                "avg_reward_100": round(avg, 2), "epsilon": round(agent.epsilon, 4),
                "loss": round(loss_val, 6), "steps": steps})
        if avg > best_reward and ep > 100:
            best_reward = avg; agent.save(model_path.replace(".pth", "_best.pth"))
        if ep % SAVE_EVERY_N_EPISODES == 0:
            agent.save(model_path.replace(".pth", "_ep%d.pth" % ep))

    agent.save(model_path)
    save_metrics(os.path.join(dirs["data"], "final_metrics.json"),
        {"scenario": scenario_name, "algorithm": "Attention-DQN", "episodes": max_episodes,
         "best_avg_reward": best_reward, "final_avg_reward": avg})
    env.close()
    print("\n训练完成。模型: %s" % model_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Attention-DQN 训练")
    p.add_argument("--scenario", type=str, default="heavy_fog", choices=DEFAULT_SCENARIOS + ["all"])
    p.add_argument("--episodes", type=int, default=MAX_EPISODES)
    p.add_argument("--render", action="store_true")
    p.add_argument("--model_dir", type=str, default="models/attention_dqn")
    args = p.parse_args()
    if args.scenario == "all":
        for s in DEFAULT_SCENARIOS:
            print("\n" + "=" * 50 + "\n  Attention-DQN: %s\n" % s + "=" * 50)
            run(s, args.episodes, args.render, args.model_dir)
    else:
        run(args.scenario, args.episodes, args.render, args.model_dir)
