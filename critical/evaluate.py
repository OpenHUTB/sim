# experiments/evaluate.py — 模型评估
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_algorithms import create_agent
from utils.data_saver import save_metrics
from utils.metrics import EpisodeStats, aggregate_episodes
from scenarios import create_scenario


def evaluate(algo_name, scenario_name, model_path, num_episodes=10,
             render=False, output_dir="results/evaluation"):
    print("=" * 50 + "\n  评估: %s x %s\n" % (algo_name, scenario_name) + "=" * 50)

    scenario = create_scenario(scenario_name)
    env = scenario.create_env()

    agent = create_agent(algo_name, state_size=env.observation_space, action_size=env.action_space)
    try:
        agent.load(model_path)
        print("模型已加载: %s" % model_path)
    except FileNotFoundError:
        print("警告: 模型不存在 %s，使用未训练权重" % model_path)

    stats_list = []
    for ep in range(1, num_episodes + 1):
        state = env.reset()
        scenario.spawn_scenario_actors(env)
        stats = EpisodeStats()
        total_reward = 0.0
        while True:
            action = agent.act(state, evaluate=True) if "dqn" in algo_name else agent.act(state, evaluate=True)[0]
            next_state, reward, done, info = env.step(action)
            stats.update(env._get_distance(), env._get_ego_speed(), abs(env._get_rel_speed()),
                         reward, done, info.get("collided", False))
            total_reward += reward; state = next_state
            scenario.step_callback(stats.total_steps)
            if render: env.render()
            if done: break
        s = stats.summary(); s["episode"] = ep; s["total_reward"] = total_reward
        stats_list.append(s)
        print("  Ep %2d | %s | Reward: %7.1f | TTC_min: %5.2fs | 危险: %d"
              % (ep, "碰撞" if s["collision"] else "安全", total_reward, s["min_ttc"], s["max_danger_level"]))

    agg = aggregate_episodes(stats_list)
    print("\n" + "-" * 50)
    print("  碰撞率: %.1f%%  安全完成率: %.1f%%  平均奖励: %.2f"
          % (agg["collision_rate"] * 100, agg["success_rate"] * 100, agg["mean_total_reward"]))
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "%s_%s.json" % (algo_name, scenario_name))
    save_metrics(save_path, {"algorithm": algo_name, "scenario": scenario_name,
                              "num_episodes": num_episodes, "summary": agg, "details": stats_list})
    env.close()
    print("结果已保存: %s" % save_path)
    return agg


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--algo", type=str, required=True, choices=["dqn","attention_dqn","ppo","smooth_ppo"])
    p.add_argument("--scenario", type=str, required=True)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--render", action="store_true")
    p.add_argument("--output_dir", type=str, default="results/evaluation")
    args = p.parse_args()
    evaluate(args.algo, args.scenario, args.model, args.episodes, args.render, args.output_dir)
