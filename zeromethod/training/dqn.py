"""Reproducible sparse-reward DQN training on GymRubiksCube."""

import argparse
import json
from time import perf_counter

import torch

from zeromethod.agents.dqn import DQNAgent
from zeromethod.environment.episode import initialize_episode
from zeromethod.environment.factory import make_environment
from zeromethod.environment.runner import EpisodeRunner
from zeromethod.environment.state_generation import derive_split_seed


def epsilon_at_step(step, *, start=1.0, end=0.05, decay_steps=50_000):
    """Linearly decay exploration from start to end over agent transitions."""
    if decay_steps <= 0:
        raise ValueError("decay_steps must be positive")
    fraction = min(max(step, 0) / decay_steps, 1.0)
    return start + fraction * (end - start)


def evaluate_agent(env, agent, *, episodes, depth, max_agent_steps, master_seed):
    """Greedy evaluation on a deterministic seed split, with no replay writes."""
    runner = EpisodeRunner(env)
    results = []
    for index in range(episodes):
        seed = derive_split_seed(master_seed, "dqn-evaluation-{}".format(index))
        results.append(runner.run(
            lambda observation: agent.act(observation, epsilon=0.0),
            depth=depth, seed=seed, max_agent_steps=max_agent_steps,
        ))
    successes = sum(result.success for result in results)
    solved_steps = [result.agent_steps for result in results if result.success]
    return {
        "episodes": episodes,
        "depth": depth,
        "successes": successes,
        "success_rate": successes / episodes,
        "mean_agent_steps": sum(r.agent_steps for r in results) / episodes,
        "mean_success_steps": sum(solved_steps) / len(solved_steps) if solved_steps else 0.0,
    }


def train_dqn(
    env, *, episodes=10_000, start_depth=1, max_depth=3,
    episodes_per_depth=2_000, max_agent_steps=20, master_seed=42,
    learning_starts=1_000, epsilon_decay_steps=50_000,
    evaluation_interval=500, evaluation_episodes=100,
    evaluation_depth=None, checkpoint_path="artifacts/dqn_v1.pt",
    device=None,
):
    """Train a sparse-reward DQN with a simple increasing-depth curriculum.

    The environment's unshaped reward is used unchanged: +1 for solved,
    otherwise 0. The agent never receives scramble actions or history.
    """
    integer_options = {
        "episodes": episodes, "start_depth": start_depth, "max_depth": max_depth,
        "episodes_per_depth": episodes_per_depth, "max_agent_steps": max_agent_steps,
        "learning_starts": learning_starts, "epsilon_decay_steps": epsilon_decay_steps,
        "evaluation_interval": evaluation_interval, "evaluation_episodes": evaluation_episodes,
        "master_seed": master_seed,
    }
    for name, value in integer_options.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("{} must be an integer".format(name))
    if episodes <= 0 or start_depth <= 0 or max_depth < start_depth:
        raise ValueError("require episodes > 0 and 1 <= start_depth <= max_depth")
    if episodes_per_depth <= 0 or max_agent_steps <= 0 or learning_starts < 0:
        raise ValueError("episode depth interval and agent step limit must be positive")
    if evaluation_interval <= 0 or evaluation_episodes <= 0:
        raise ValueError("evaluation settings must be positive")
    if evaluation_depth is None:
        evaluation_depth = max_depth
    if isinstance(evaluation_depth, bool) or not isinstance(evaluation_depth, int) or evaluation_depth <= 0:
        raise ValueError("evaluation_depth must be a positive integer")

    agent = DQNAgent(seed=master_seed, device=device)
    global_step = 0
    losses = []
    started = perf_counter()
    last_evaluation = None

    for episode_index in range(episodes):
        depth = min(max_depth, start_depth + episode_index // episodes_per_depth)
        scramble_seed = derive_split_seed(master_seed, "dqn-training-{}".format(episode_index))
        state, _ = initialize_episode(env, depth=depth, seed=scramble_seed)
        episode_reward = 0.0
        episode_steps = 0
        solved = env.is_solved()

        while not solved and episode_steps < max_agent_steps:
            epsilon = epsilon_at_step(global_step, decay_steps=epsilon_decay_steps)
            action = agent.act(state, epsilon=epsilon)
            transition = env.step(action)
            next_state = transition.observation
            done = transition.terminated or transition.truncated
            agent.remember(state, action, transition.reward, next_state, done)
            episode_reward += transition.reward
            episode_steps += 1
            global_step += 1
            state = next_state
            solved = transition.terminated

            if global_step >= learning_starts:
                loss = agent.learn()
                if loss is not None:
                    losses.append(loss)

            if done:
                break

        if (episode_index + 1) % evaluation_interval == 0 or episode_index + 1 == episodes:
            last_evaluation = evaluate_agent(
                env, agent, episodes=evaluation_episodes,
                depth=evaluation_depth, max_agent_steps=max_agent_steps,
                master_seed=derive_split_seed(master_seed, "dqn-validation-{}".format(episode_index)),
            )
            print(json.dumps({
                "episode": episode_index + 1,
                "curriculum_depth": depth,
                "global_steps": global_step,
                "epsilon": epsilon_at_step(global_step, decay_steps=epsilon_decay_steps),
                "episode_reward": episode_reward,
                "episode_steps": episode_steps,
                "replay_size": len(agent.replay_buffer),
                "mean_recent_loss": sum(losses[-500:]) / len(losses[-500:]) if losses else None,
                "validation": last_evaluation,
            }, sort_keys=True), flush=True)
            agent.save(checkpoint_path, metadata={
                "algorithm": "DQN", "master_seed": master_seed,
                "episode": episode_index + 1, "global_steps": global_step,
                "start_depth": start_depth, "max_depth": max_depth,
                "episodes_per_depth": episodes_per_depth,
                "max_agent_steps": max_agent_steps,
                "evaluation": last_evaluation,
            })

    return {
        "algorithm": "DQN",
        "episodes": episodes,
        "global_steps": global_step,
        "replay_size": len(agent.replay_buffer),
        "training_updates": agent.training_steps,
        "mean_loss": sum(losses) / len(losses) if losses else None,
        "last_validation": last_evaluation,
        "elapsed_seconds": perf_counter() - started,
        "checkpoint": str(checkpoint_path),
        "device": str(agent.device),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=10_000)
    parser.add_argument("--start-depth", type=int, default=1)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--episodes-per-depth", type=int, default=2_000)
    parser.add_argument("--max-agent-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--epsilon-decay-steps", type=int, default=50_000)
    parser.add_argument("--evaluation-interval", type=int, default=500)
    parser.add_argument("--evaluation-episodes", type=int, default=100)
    parser.add_argument("--evaluation-depth", type=int, default=None)
    parser.add_argument("--checkpoint", default="artifacts/dqn_v1.pt")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    env = make_environment(max_steps=None)
    try:
        report = train_dqn(
            env, episodes=args.episodes, start_depth=args.start_depth,
            max_depth=args.max_depth, episodes_per_depth=args.episodes_per_depth,
            max_agent_steps=args.max_agent_steps, master_seed=args.seed,
            learning_starts=args.learning_starts,
            epsilon_decay_steps=args.epsilon_decay_steps,
            evaluation_interval=args.evaluation_interval,
            evaluation_episodes=args.evaluation_episodes,
            evaluation_depth=args.evaluation_depth,
            checkpoint_path=args.checkpoint, device=args.device,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    finally:
        env.close()


if __name__ == "__main__":
    main()
