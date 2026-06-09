# experiments/run_ppo.py — 标准 PPO 训练入口
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_algorithms.ppo import PPOAgent
from utils.data_saver import init_result_dir, save_training_log, save_model, save_metrics
from utils.metrics import MovingAverage
from config.ppo_config import MAX_EPISODES, SAVE_EVERY_N_EPISODES, LOG_EVERY_N_EPISODES
from scenarios import create_scenario

DEFAULT_SCENARIOS = ["rain_storm", "pedestrian_cross", "jaywalking"]


def run(scenario_name="rain_storm", max_episodes=MAX_EPISODES,
        render=False, model_dir="models/ppo"):
    scenario = create_scenario(scenario_name)
    env = scenario.create_env()
    print("场景: %s | 类别: %s" % (scenario.name, scenario.category))

    agent = PPOAgent(state_size=env.observation_space, action_size=env.action_space)
    print("智能体: PPO | update_every: %d | k_epochs: %d" % (agent.update_every, agent.k_epochs))

    dirs = init_result_dir("ppo_%s" % scenario_name, base_dir="results/ppo")
    model_path = os.path.join(model_dir, "ppo_%s.pth" % scenario_name)
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
            action, log_prob = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.store(state, action, log_prob, reward, next_state, done)
            agent.train(); agent.total_steps += 1
            scenario.step_callback(steps)
            state = next_state; episode_reward += reward; steps += 1
            if render and ep % 20 == 0: env.render()
            if done: break
        agent.end_episode(episode_reward)
        reward_ma.update(episode_reward); avg = reward_ma.mean()
        if ep % LOG_EVERY_N_EPISODES == 0:
            li = agent.last_loss_info or {}
            print("Ep %4d | Reward: %7.1f | Avg100: %7.1f | A_loss: %.4f | C_loss: %.4f"
                  % (ep, episode_reward, avg, li.get("actor_loss", 0), li.get("critic_loss", 0)))
            save_training_log(log_path, {"episode": ep, "reward": round(episode_reward, 2),
                "avg_reward_100": round(avg, 2), "actor_loss": round(li.get("actor_loss", 0), 6),
                "critic_loss": round(li.get("critic_loss", 0), 6), "steps": steps})
        if avg > best_reward and ep > 100:
            best_reward = avg; agent.save(model_path.replace(".pth", "_best.pth"))
        if ep % SAVE_EVERY_N_EPISODES == 0:
            agent.save(model_path.replace(".pth", "_ep%d.pth" % ep))

    agent.save(model_path)
    save_metrics(os.path.join(dirs["data"], "final_metrics.json"),
        {"scenario": scenario_name, "algorithm": "PPO", "episodes": max_episodes,
         "best_avg_reward": best_reward, "final_avg_reward": avg})
    env.close()
    print("\n训练完成。模型: %s" % model_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="PPO 训练")
    p.add_argument("--scenario", type=str, default="rain_storm", choices=DEFAULT_SCENARIOS + ["all"])
    p.add_argument("--episodes", type=int, default=MAX_EPISODES)
    p.add_argument("--render", action="store_true")
    p.add_argument("--model_dir", type=str, default="models/ppo")
    args = p.parse_args()
    if args.scenario == "all":
        for s in DEFAULT_SCENARIOS:
            print("\n" + "=" * 50 + "\n  PPO: %s\n" % s + "=" * 50)
            run(s, args.episodes, args.render, args.model_dir)
    else:
        run(args.scenario, args.episodes, args.render, args.model_dir)
