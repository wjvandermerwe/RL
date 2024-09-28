import os
import random
import numpy as np
import torch
import gym
from dqn.agent import DQNAgent
from dqn.wrappers import *
from dqn.helpers import create_pong_gif

if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    env = gym.make("PongNoFrameskip-v4", render_mode='rgb_array')
    env.seed(seed)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    env = FireResetEnv(env)
    env = ClipRewardEnv(env)
    env = WarpFrame(env)
    env = PyTorchFrame(env)
    env = FrameStack(env, k=5)

    agent = DQNAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        replay_buffer=None,  # no replay buffer during inference
        use_double_dqn=True,
        lr=1e-4,
        batch_size=32,
        gamma=0.99,
    )

    checkpoint_path = "./checkpoints/checkpoint_426_900000.pth"
    if os.path.exists(checkpoint_path):
        step, episode = agent.load_checkpoint(checkpoint_path)
        print(f"Loaded checkpoint from step {step}, episode {episode}")
    else:
        print(f"Checkpoint {checkpoint_path} not found.")
        exit()

    create_pong_gif(
        env,
        agent=agent,
        gif_path="pong_agent_inference.gif",
        num_episodes=1
    )
