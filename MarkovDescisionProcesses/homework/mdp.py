import numpy as np
import random
import matplotlib.pyplot as plt

class GridworldMDP:
    def __init__(self, grid_size=7):
        self.grid_size = grid_size
        self.initial_state = (grid_size - 1, 0)
        self.goal_state = (0, 0)
        self.obstacles = [(2, i) for i in range(grid_size - 1)]
        self.state = self.initial_state

    def reset(self):
        self.state = self.initial_state
        return self.state

    def step(self, action):
        if self.state == self.goal_state:
            return self.state, 0, True

        next_state = self.next_state(self.state, action)

        if next_state in self.obstacles or not self.is_valid_state(next_state):
            next_state = self.state

        reward = 20 if next_state == self.goal_state else -1
        done = next_state == self.goal_state
        self.state = next_state

        return next_state, reward, done

    def next_state(self, state, action):
        if action == 'up':
            return (state[0] - 1, state[1])
        elif action == 'down':
            return (state[0] + 1, state[1])
        elif action == 'left':
            return (state[0], state[1] - 1)
        elif action == 'right':
            return (state[0], state[1] + 1)

    def is_valid_state(self, state):
        return 0 <= state[0] < self.grid_size and 0 <= state[1] < self.grid_size


def value_iteration(env, gamma=1.0, theta=0.0001):
    value_function = np.zeros((env.grid_size, env.grid_size))
    actions = ['up', 'down', 'left', 'right']
    delta = float('inf')

    while delta > theta:
        delta = 0
        for i in range(env.grid_size):
            for j in range(env.grid_size):
                state = (i, j)
                if state == env.goal_state or state in env.obstacles:
                    continue

                v = value_function[state]
                max_value = float('-inf')

                for action in actions:
                    next_state = env.next_state(state, action)
                    if next_state in env.obstacles or not env.is_valid_state(next_state):
                        next_state = state

                    reward = 20 if next_state == env.goal_state else -1
                    value = reward + gamma * value_function[next_state]

                    if value > max_value:
                        max_value = value

                value_function[state] = max_value
                delta = max(delta, abs(v - max_value))

    return value_function


def greedy_policy(state, optimal_value, env):
    actions = ['up', 'down', 'left', 'right']
    best_action = None
    best_value = float('-inf')

    for action in actions:
        next_state = env.next_state(state, action)
        if next_state in env.obstacles or not env.is_valid_state(next_state):
            next_state = state

        value = optimal_value[next_state]
        if value > best_value:
            best_value = value
            best_action = action

    return best_action


def run_random_agent(env, max_steps=50):
    actions = ['up', 'down', 'left', 'right']
    state = env.reset()
    total_reward = 0
    trajectory = []

    for _ in range(max_steps):
        action = random.choice(actions)
        next_state, reward, done = env.step(action)
        total_reward += reward
        trajectory.append(state)
        state = next_state

        if done:
            break

    return total_reward, trajectory

def run_greedy_agent(env, value_function, max_steps=50):
    state = env.reset()
    total_reward = 0
    trajectory = []

    for _ in range(max_steps):
        if state == env.goal_state:
            break
        action = greedy_policy(state, value_function, env)
        next_state, reward, done = env.step(action)
        total_reward += reward
        trajectory.append(state)
        state = next_state

    return total_reward, trajectory

def collect_rewards(agent_fn, env, value_function=None, runs=20, max_steps=50):
    all_rewards = []
    all_trajectories = []

    for _ in range(runs):
        if agent_fn == run_greedy_agent:
            reward, trajectory = agent_fn(env, value_function, max_steps)
        else:
            reward, trajectory = agent_fn(env, max_steps)
        all_rewards.append(reward)
        all_trajectories.append(trajectory)

    return all_rewards, all_trajectories


grid = GridworldMDP()
optimal_value = value_iteration(grid)

greed_rewards, greed_trajectories = collect_rewards(run_greedy_agent, grid, optimal_value)
rand_rewards, rand_trajectories = collect_rewards(run_random_agent, grid, optimal_value)



avg_rewards_rand = np.mean(rand_rewards, axis=0)
avg_rewards_greed = np.mean(greed_rewards, axis=0)

labels = ['Random Agent', 'Greedy Agent']
avg_rewards = [avg_rewards_rand, avg_rewards_greed]

plt.figure(figsize=(10, 5))
plt.bar(labels, avg_rewards, color=['blue', 'green'])
plt.xlabel('Agent')
plt.ylabel('Average Return')
plt.title('Average Returns of Random and Greedy Agents over 20 Runs')
plt.show()

# plt.savefig('barchart.png')
def visualize_trajectory(trajectory, env):
    grid = np.zeros((env.grid_size, env.grid_size))

    for obstacle in env.obstacles:
        grid[obstacle] = -1

    grid[env.goal_state] = 2

    for (x, y) in trajectory:
        grid[x, y] = 1

    grid[trajectory[-1]] = 3

    return grid


random_trajectory_grid = visualize_trajectory(rand_trajectories[0], grid)
greedy_trajectory_grid = visualize_trajectory(greed_trajectories[0], grid)

fig, axs = plt.subplots(1, 2, figsize=(15, 7))

axs[0].imshow(random_trajectory_grid, cmap='gray', interpolation='nearest')
axs[0].set_title('Random Agent Trajectory')

axs[1].imshow(greedy_trajectory_grid, cmap='gray', interpolation='nearest')
axs[1].set_title('Greedy Agent Trajectory')

plt.show()
# plt.savefig('trajectories.png')