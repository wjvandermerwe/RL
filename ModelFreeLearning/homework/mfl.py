

import numpy as np
import gym
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

env = gym.make('CliffWalking-v0')
alpha = 0.5
epsilon = 0.1
lambdas = [0, 0.3, 0.5]
num_episodes = 200
num_runs = 100

# useful: https://github.com/openai/gym/blob/dcd185843a62953e27c2d54dc8c2d647d604b635/gym/envs/toy_text/cliffwalking.py
def e_greedy_policy(Q, state, epsilon, action_space):
    num_actions = action_space
    policy = np.ones(num_actions) * epsilon / num_actions
    best_action = np.argmax(Q[state])
    policy[best_action] += (1.0 - epsilon)
    action = np.random.choice(np.arange(num_actions), p=policy)
    return action

def sarsa_lambda(env, alpha, epsilon, lambd, num_episodes):
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    returns = []
    frames = []
    for episode in range(num_episodes):
        state, _ = env.reset()

        eligibility_trace = np.zeros_like(q_table)
        action = e_greedy_policy(q_table, state, epsilon, env.action_space.n)
        total_return = 0
        while True:
            next_state, reward, done, _, _ = env.step(action)
            total_return += reward
            next_action = e_greedy_policy(q_table, next_state, epsilon, env.action_space.n)
            td_error = reward + q_table[next_state, next_action] - q_table[state, action]
            eligibility_trace[state, action] += 1
            q_table += alpha * td_error * eligibility_trace
            eligibility_trace *= lambd
            state, action = next_state, next_action
            if done:
                break
        returns.append(total_return)

        nrow, ncol = env.shape
        heatmap = np.max(q_table, axis=1).reshape(nrow, ncol)
        frames.append(heatmap)

    return q_table, returns, frames


all_frames = []
all_returns = []
for lambd in lambdas:
    q_table, returns, frames = sarsa_lambda(env, alpha, epsilon, lambd, num_episodes)
    all_frames.extend(frames)
    all_returns.append(returns)
fig, axs = plt.subplots(1, len(lambdas), figsize=(15, 5))  # Create one row with multiple subplots


def update(frame_set):
    for i in range(len(lambdas)):
        axs[i].imshow(frame_set[i], cmap='coolwarm')
        axs[i].set_title(f'λ={lambdas[i]}')




ani = FuncAnimation(fig, update, frames=[all_frames[i:i + len(lambdas)] for i in range(0, len(all_frames), len(lambdas))], blit=False, repeat=False)




ani.save('sarsa_lambda_value_function_animation.gif', writer='pillow', fps=4)
plt.figure()

for lambd, avg_returns in zip(lambdas, all_returns):
    plt.plot(avg_returns, label=f'λ={lambd}')
    plt.fill_between(range(num_episodes), avg_returns - np.std(avg_returns), avg_returns + np.std(avg_returns), alpha=0.2)
plt.title("Average Return over Time for Different λ")
plt.xlabel("Episode")
plt.ylabel("Average Return")
plt.legend()
plt.savefig("avg_return.png")
