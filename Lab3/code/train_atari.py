import random

from dqn.helpers import plot_reward_curve, plot_loss_curve, create_pong_gif, plot_action_value_curve
from dqn.agent import DQNAgent
from dqn.replay_buffer import ReplayBuffer
from dqn.wrappers import *

if __name__ == "__main__":

    hyper_params = {
        "seed": 42,
        "env": "PongNoFrameskip-v4",
        "replay-buffer-size": int(5e3),
        "learning-rate": 1e-4,
        "discount-factor": 0.99,
        "num-steps": int(1e6),
        "batch-size": 256,
        "learning-starts": 10000,
        "learning-freq": 5,
        "use-double-dqn": True,
        "target-update-freq": 1000,
        "eps-start": 1.0,
        "eps-end": 0.01,
        "eps-fraction": 0.1,
        "print-freq": 10,
    }

    np.random.seed(hyper_params["seed"])
    random.seed(hyper_params["seed"])

    assert "NoFrameskip" in hyper_params["env"], "Require environment with no frameskip"
    env = gym.make(hyper_params["env"])
    env.seed(hyper_params["seed"])

    # Apply wrappers
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    env = FireResetEnv(env)
    env = ClipRewardEnv(env)
    env = WarpFrame(env)
    env = PyTorchFrame(env)
    env = FrameStack(env, k=5)

    replay_buffer = ReplayBuffer(hyper_params["replay-buffer-size"])

    agent = DQNAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        replay_buffer=replay_buffer,
        batch_size=hyper_params["batch-size"],
        lr=hyper_params["learning-rate"],
        gamma=hyper_params["discount-factor"],
    )

    current_episode_action_values  = []
    eps_timesteps = hyper_params["eps-fraction"] * float(hyper_params["num-steps"])
    episode_rewards = [0.0]
    episode_action_values = []
    current_episode_rewards = 0.0

    losses = []
    episode_num = 0

    state = env.reset()
    for t in range(hyper_params["num-steps"]):
        fraction = min(1.0, float(t) / eps_timesteps)
        eps_threshold = hyper_params["eps-start"] + fraction * (
            hyper_params["eps-end"] - hyper_params["eps-start"]
        )
        sample = random.random()

        action, q_value = agent.act(state, eps_threshold)

        next_state, reward, done, info = env.step(action)
        replay_buffer.add(state, action, reward, next_state, float(done))
        state = next_state

        current_episode_rewards += reward
        current_episode_action_values.append(q_value)

        if done:
            state = env.reset()
            episode_rewards.append(current_episode_rewards)
            average_action_value = np.mean(current_episode_action_values)
            episode_action_values.append(average_action_value)
            current_episode_rewards = 0.0
            current_episode_action_values = []
            episode_num += 1

        if (
            t > hyper_params["learning-starts"]
            and t % hyper_params["learning-freq"] == 0
        ):
            loss = agent.optimise_td_loss()
            losses.append(loss)

        if (
            t > hyper_params["learning-starts"]
            and t % hyper_params["target-update-freq"] == 0
        ):
            agent.update_target_network()

        num_episodes = len(episode_rewards)

        if t % 100000 == 0:
            agent.save_checkpoint(step=t, episode=num_episodes)

        if (
            done
            and hyper_params["print-freq"] is not None
            and len(episode_rewards) % hyper_params["print-freq"] == 0
        ):
            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            print("********************************************************")
            print(f"steps: {t}")
            print(f"episodes: {num_episodes}")
            print(f"mean 100 episode reward: {mean_100ep_reward}")
            print(f"% time spent exploring: {int(100 * eps_threshold)}")
            print("********************************************************")

    plot_reward_curve(episode_rewards, save_path="reward_curve.png")
    plot_action_value_curve(episode_action_values, save_path="action_value_curve.png")
    plot_loss_curve(losses, save_path="loss_curve.png")
