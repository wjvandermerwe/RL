import matplotlib.pyplot as plt
import numpy as np
import imageio
import gym
from pathlib import Path


def plot_reward_curve(episode_rewards, save_path="reward_curve.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(episode_rewards)), episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Average Score per Episode")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved reward curve at: {save_path}")

def plot_action_value_curve(episode_action_values, save_path="action_value_curve.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(episode_action_values)), episode_action_values)
    plt.xlabel("Episode")
    plt.ylabel("Average Action Value")
    plt.title("Average Action Value per Episode")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved action value curve at: {save_path}")

def plot_loss_curve(losses, save_path="loss_curve.png"):
    """
    Plot the loss curve during training and save the plot.

    :param losses: List of loss values per optimization step.
    :param save_path: Path to save the plot.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(losses)), losses)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Loss Curve During Training")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved loss curve at: {save_path}")


def create_pong_gif(env, agent=None, gif_path="pong_agent_inference.gif", num_episodes=1):

    frames = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            frame = env.env.render(mode='rgb_array')  # Access the base environment
            frames.append(frame)
            action = agent.act(state, eps=0.0) if agent else env.action_space.sample()
            state, _, done, _ = env.step(action)

    env.close()

    Path(gif_path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(gif_path, frames, fps=30)
    print(f"Saved GIF of agent playing Pong at: {gif_path}")
