import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

class GridworldMDP:
    def __init__(self, grid_size=4):
        self.grid_size = grid_size
        self.initial_state = (grid_size - 1, 0)
        self.goal_state = (0, 0)
        self.state = self.initial_state

    def reset(self):
        self.state = self.initial_state
        return self.state

    def step(self, action):
        if self.state == self.goal_state:
            return self.state, 0, True

        next_state = self.next_state(self.state, action)

        if not self.is_valid_state(next_state):
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


def policy_evaluation_inplace(env, policy, gamma=1.0, theta=0.01):
    value_function = np.zeros((env.grid_size, env.grid_size))
    delta = float('inf')
    iterations=0
    while delta >= theta:
        delta = 0
        iterations +=1
        for i in range(env.grid_size):
            for j in range(env.grid_size):
                state_index = (i, j)
                if state_index == env.goal_state:
                    continue

                v = value_function[state_index]
                new_value = 0

                for action, action_prob in policy[state_index].items():
                    next_state = env.next_state(state_index, action)
                    if not env.is_valid_state(next_state):
                        next_state = state_index
                    reward = 20 if next_state == env.goal_state else -1

                    # update the value for the current state
                    new_value += action_prob * (reward + gamma * value_function[next_state])

                value_function[state_index] = new_value
                delta = max(delta, abs(v - new_value))

    return value_function, iterations


def policy_evaluation_2array(env, policy, gamma=1.0, theta=0.01):
    old_values = np.zeros((env.grid_size, env.grid_size))
    new_values = np.zeros_like(old_values)
    delta = float('inf')
    iterations = 0
    while delta >= theta:
        delta = 0
        iterations += 1
        for i in range(env.grid_size):
            for j in range(env.grid_size):
                state_index = (i, j)
                if state_index == env.goal_state:
                    continue

                new_value = 0
                for action, action_prob in policy[state_index].items():
                    next_state = env.next_state(state_index, action)
                    if not env.is_valid_state(next_state):
                        next_state = state_index

                    reward = 20 if next_state == env.goal_state else -1
                    new_value += action_prob * (reward + gamma * old_values[next_state])

                new_values[state_index] = new_value
                delta = max(delta, abs(old_values[state_index] - new_value))

        # Copy the new values into the old values array for the next iteration
        old_values = new_values.copy()

    return new_values, iterations

# given unified random policy
policy = {
    (i, j): {'up': 0.25, 'down': 0.25, 'left': 0.25, 'right': 0.25}
    for i in range(4) for j in range(4)
}
# dont move in goal state
policy[(0, 0)] = {}



grid = GridworldMDP()
optimal_values, _ = policy_evaluation_2array(grid, policy)
sns.heatmap(optimal_values, annot=True, cmap="YlGnBu", cbar=True)
plt.title("Optimal Value Function")

discount_values = np.logspace(-0.2, 0, num=20)
twoarray = []
inplace = []
for discount in discount_values:
    _, iterations = policy_evaluation_2array(grid, policy, gamma=discount)
    twoarray.append(iterations)
    _, iterations = policy_evaluation_inplace(grid, policy, gamma=discount)
    inplace.append(iterations)

# plt.savefig('heatmap.png')



plt.figure(figsize=(10, 6))
plt.plot(discount_values, twoarray, marker='o', label='Two Array')
plt.plot(discount_values, inplace, marker='x', label='Inplace')

plt.xlabel('Discount Factor (γ)')
plt.ylabel('Iterations Until Convergence')
plt.title('Iterations Until Convergence vs. Discount Factor')
plt.legend()
plt.grid(True)
# plt.savefig('convergence.png')





