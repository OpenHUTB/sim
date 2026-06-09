# tests/test_agent.py
# 测试所有 4 种 RL 智能体是否能正常初始化、决策、训练

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from rl_algorithms import ALGORITHM_REGISTRY, BaseAgent
from config.dqn_config import STATE_SIZE as DQN_STATE, ACTION_SIZE as DQN_ACTION
from config.ppo_config import STATE_SIZE as PPO_STATE, ACTION_SIZE as PPO_ACTION


def test_agent_init(algo_name):
    """测试智能体初始化"""
    print("  >>> 测试初始化: %s" % algo_name)
    try:
        state_size = DQN_STATE if "dqn" in algo_name else PPO_STATE
        action_size = DQN_ACTION if "dqn" in algo_name else PPO_ACTION
        agent = ALGORITHM_REGISTRY[algo_name](state_size=state_size,
                                              action_size=action_size)
        assert agent.state_size == state_size, "state_size 不匹配"
        assert agent.action_size == action_size, "action_size 不匹配"
        assert agent.name, "智能体名称缺失"
        print("    PASS: %s 初始化成功 (device=%s)"
              % (agent.name, agent.device))
        return agent
    except Exception as e:
        print("    FAIL: %s" % e)
        return None


def test_agent_act(agent, algo_name):
    """测试智能体决策（随机输入）"""
    print("  >>> 测试决策: %s" % algo_name)
    try:
        state_size = agent.state_size
        state = np.random.randn(state_size).astype(np.float32)

        if "dqn" in algo_name:
            action = agent.act(state, evaluate=False)
            # 探索模式下也可能随机
            action_greedy = agent.act(state, evaluate=True)
            assert 0 <= action < agent.action_size, \
                "动作越界: %d" % action
            assert 0 <= action_greedy < agent.action_size, \
                "贪心动作越界: %d" % action_greedy
            print("    PASS: act 正常 (explore=%d, greedy=%d)"
                  % (action, action_greedy))
        else:
            action, log_prob = agent.act(state, evaluate=False)
            action_g, log_prob_g = agent.act(state, evaluate=True)
            assert 0 <= action < agent.action_size, \
                "动作越界: %d" % action
            assert isinstance(log_prob, float), \
                "log_prob 类型错误: %s" % type(log_prob)
            print("    PASS: act 正常 (action=%d, log_prob=%.4f)"
                  % (action, log_prob))
        return True
    except Exception as e:
        print("    FAIL: %s" % e)
        return False


def test_agent_train(agent, algo_name):
    """测试智能体训练（需要先存储一些 transition）"""
    print("  >>> 测试训练: %s" % algo_name)
    state_size = agent.state_size
    action_size = agent.action_size

    try:
        if "dqn" in algo_name:
            # 预填充回放池
            for _ in range(3000):
                s = np.random.randn(state_size).astype(np.float32)
                a = np.random.randint(action_size)
                r = np.random.randn()
                ns = np.random.randn(state_size).astype(np.float32)
                d = np.random.random() < 0.1
                agent.store(s, a, r, ns, d)
            loss_info = agent.train()
            print("    PASS: train 完成 (loss=%s)"
                  % (loss_info["loss"] if loss_info else "None"))
        else:
            # PPO 系需要填充 rollout storage
            for _ in range(2048):
                s = np.random.randn(state_size).astype(np.float32)
                a = np.random.randint(action_size)
                lp = -np.log(action_size)  # 均匀分布的 log_prob
                r = np.random.randn()
                ns = np.random.randn(state_size).astype(np.float32)
                d = False
                agent.store(s, a, lp, r, ns, d)
                agent.total_steps += 1
            loss_info = agent.train()
            if loss_info:
                print("    PASS: train 完成 (actor_loss=%.4f critic_loss=%.4f)"
                      % (loss_info.get("actor_loss", 0),
                         loss_info.get("critic_loss", 0)))
            else:
                print("    PASS: train 返回 None (未达到 update_every)")
        return True
    except Exception as e:
        print("    FAIL: %s" % e)
        import traceback
        traceback.print_exc()
        return False


def test_agent_save_load(agent, algo_name, tmp_path="/tmp/test_agent.pth"):
    """测试模型保存与加载"""
    print("  >>> 测试持久化: %s" % algo_name)
    try:
        agent.save(tmp_path)
        assert os.path.exists(tmp_path), "保存失败: 文件不存在"

        # 创建新智能体并加载
        new_agent = ALGORITHM_REGISTRY[algo_name](
            state_size=agent.state_size, action_size=agent.action_size)
        new_agent.load(tmp_path)

        print("    PASS: save/load 正常")
        os.remove(tmp_path)
        return True
    except Exception as e:
        print("    FAIL: %s" % e)
        return False


def run_all():
    """运行全部智能体测试"""
    print("=" * 50)
    print("  智能体测试套件 (%d 算法)" % len(ALGORITHM_REGISTRY))
    print("=" * 50)

    results = {}

    for algo_name in ALGORITHM_REGISTRY:
        print("\n—" * 25)
        print("  测试: %s" % algo_name)
        agent = test_agent_init(algo_name)
        results["%s_init" % algo_name] = agent is not None

        if agent is not None:
            results["%s_act" % algo_name] = test_agent_act(agent, algo_name)
            results["%s_train" % algo_name] = test_agent_train(agent, algo_name)
            results["%s_save" % algo_name] = test_agent_save_load(agent, algo_name)

    print("\n" + "=" * 50)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print("  结果: %d/%d 通过" % (passed, total))
    for name, ok in results.items():
        if not ok:
            print("    失败: %s" % name)
    print("=" * 50)

    return passed == total


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="智能体测试")
    parser.add_argument("--algo", type=str, default=None,
                        choices=list(ALGORITHM_REGISTRY.keys()),
                        help="测试单个算法")
    args = parser.parse_args()

    if args.algo:
        agent = test_agent_init(args.algo)
        if agent:
            test_agent_act(agent, args.algo)
            test_agent_train(agent, args.algo)
            test_agent_save_load(agent, args.algo)
    else:
        success = run_all()
        sys.exit(0 if success else 1)
