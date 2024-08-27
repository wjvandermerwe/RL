###
# Group Members
# Willem:2914429
###

import numpy as np
from torch.utils.hipify.hipify_python import value

from DynamicProgramming.homework.mdp import policy
from environments.gridworld import GridworldEnv, UP, DOWN, RIGHT, LEFT
import timeit
import matplotlib.pyplot as plt


def policy_evaluation(env: GridworldEnv, policy, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.

    Args:

        env: OpenAI environment.
            env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.observation_space.n is a number of states in the environment.
            env.action_space.n is a number of actions in the environment.
        policy: [S, A] shaped matrix representing the policy.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        Vector of length env.observation_space.n representing the value function.
    """
    value_function = np.zeros(env.observation_space.n)
    delta = float('inf')
    while delta > theta:
        delta = 0
        for s in range(env.observation_space.n):
            v = value_function[s]
            new_v=0
            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in env.P[s][a]:
                    # joint probability of action taken and environment change to be made
                    new_v += action_prob * prob * (reward + discount_factor * value_function[next_state])
            value_function[s] = new_v
            delta = max(delta, abs(v - value_function[s]))
    return value_function


def policy_iteration(env, policy_evaluation_fn=policy_evaluation, discount_factor=1.0):
    """
    Iteratively evaluates and improves a policy until an optimal policy is found.

    Args:
        env: The OpenAI environment.
        policy_evaluation_fn: Policy Evaluation function that takes 3 arguments:
            env, policy, discount_factor.
        discount_factor: gamma discount factor.

    Returns:
        A tuple (policy, V).
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.

    """
    value_function = np.ones(env.observation_space.n)
    init_policy = [[np.random.random() for _ in range(env.action_space.n)] for _ in range(env.observation_space.n)]
    # norm
    policy = [[val / sum(init_policy[i]) for val in init_policy[i]] for i in range(env.observation_space.n)]
    value_function = policy_evaluation_fn(env, policy)

    def one_step_lookahead(state, V):
        """
        Helper function to calculate the value for all action in a given state.

        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.observation_space.n

        Returns:
            A vector of length env.action_space.n containing the expected value of each action.
        """
        A = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            for prob, next_state, reward, done in env.P[state][a]:
                A[a] += prob * (reward + discount_factor * V[next_state])
        return A

    loop = True
    while loop:
        policy_stable = True
        for s in range(env.observation_space.n):
            old = np.argmax(policy[s])
            best_a = np.argmax(one_step_lookahead(s, value_function))
            # useful trick -> pick row corresponding to best action and get identity entry
            policy[s] = np.eye(env.action_space.n)[best_a]
            if old != best_a:
                policy_stable = False
        if policy_stable:
            loop = False
        else:
            value_function = policy_evaluation_fn(env, policy)

    return (policy, value_function)

def value_iteration(env, theta=0.0001, discount_factor=1.0):
    """
    Value Iteration Algorithm.

    Args:
        env: OpenAI environment.
            env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.observation_space.n is a number of states in the environment.
            env.action_space.n is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.
    """

    def one_step_lookahead(state, V):
        """
        Helper function to calculate the value for all action in a given state.

        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.observation_space.n

        Returns:
            A vector of length env.action_space.n containing the expected value of each action.
        """
        A = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            for prob, next_state, reward, done in env.P[state][a]:
                A[a] += prob * (reward + discount_factor * V[next_state])

        return A

    value_function = np.ones(env.observation_space.n)
    value_function[24] = 0
    loop = True
    while loop:
        delta = 0
        for s in range(env.observation_space.n):
            v = value_function[s]
            value_function[s] = np.max(one_step_lookahead(s, value_function))
            delta = max(delta, abs(v - value_function[s]))
        if delta < theta:
            loop = False

    policy = [np.eye(env.action_space.n)[np.argmax(one_step_lookahead(s, value_function))] for s in range(env.observation_space.n)]
    return policy, value_function


action_mapping = {UP: 'U', RIGHT: 'R', DOWN: 'D', LEFT: 'L'}

def main():
    # Create Gridworld environment with size of 5 by 5, with the goal at state 24. Reward for getting to goal state is 0, and each step reward is -1
    env = GridworldEnv(shape=[5, 5], terminal_states=[
                       24], terminal_reward=0, step_reward=-1)
    state = env.reset()
    print("")
    env.render()
    print("")

    # 1.1
    grid = np.full(env.shape, 'o', dtype='<U1')
    start = state
    # trajectory = []
    done = False
    while not done:
        # uniform by default
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        # trajectory.append((state, action))
        y, x = np.unravel_index(state, env.shape)
        grid[y, x] = action_mapping[action]
        state = next_state


    # 1.2
    start_y, start_x = np.unravel_index(start, env.shape)
    grid[start_y, start_x] = 'S'
    final_y, final_x = np.unravel_index(state, env.shape)
    grid[final_y, final_x] = 'X'
    for row in grid:
        print(" ".join(row))

    print("*" * 5 + " Policy evaluation " + "*" * 5)
    print("")
    state = env.reset()
    # 2.1
    policy = [[0.25, 0.25, 0.25, 0.25] for _ in range(env.observation_space.n)]
    v = policy_evaluation(env, policy=policy)

    print(v)

    # Test: Make sure the evaluated policy is what we expected
    expected_v = np.array([-106.81, -104.81, -101.37, -97.62, -95.07,
                           -104.81, -102.25, -97.69, -92.40, -88.52,
                           -101.37, -97.69, -90.74, -81.78, -74.10,
                           -97.62, -92.40, -81.78, -65.89, -47.99,
                           -95.07, -88.52, -74.10, -47.99, 0.0])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=2)

    env.reset()
    print("*" * 5 + " Policy iteration " + "*" * 5)
    print("")
    # 3.1
    policy, v = policy_iteration(env, policy_evaluation)

    print([np.argmax(state_actions) for state_actions in policy])
    print(v)
    # Test: Make sure the value function is what we expected
    expected_v = np.array([-8., -7., -6., -5., -4.,
                           -7., -6., -5., -4., -3.,
                           -6., -5., -4., -3., -2.,
                           -5., -4., -3., -2., -1.,
                           -4., -3., -2., -1., 0.])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=1)

    print("*" * 5 + " Value iteration " + "*" * 5)
    print("")
    # 4.1
    policy, v = value_iteration(env)  # call value_iteration

    print([np.argmax(state_actions) for state_actions in policy])
    print(v)

    # Test: Make sure the value function is what we expected
    expected_v = np.array([-8., -7., -6., -5., -4.,
                           -7., -6., -5., -4., -3.,
                           -6., -5., -4., -3., -2.,
                           -5., -4., -3., -2., -1.,
                           -4., -3., -2., -1., 0.])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=1)

    # 4.2 a
    discounts = np.logspace(-0.2, 0, num=30)

    # 4.2 b
    a_times = []
    b_times = []

    a_plots = []
    b_plots = []
    for discount in discounts:
        for _ in range(10):
            start_time = timeit.default_timer()
            policy_iteration(env, policy_evaluation, discount)
            end_time = timeit.default_timer()
            runtime = end_time - start_time
            a_times.append(runtime)
            start_time = timeit.default_timer()
            policy_iteration(env, policy_evaluation, discount)
            end_time = timeit.default_timer()
            runtime = end_time - start_time
            b_times.append(runtime)

        a_plots.append((discount,np.mean(a_times)))
        b_plots.append((discount,np.mean(b_times)))

    a_discounts, a_avg_times = zip(*a_plots)
    b_discounts, b_avg_times = zip(*b_plots)

    plt.figure(figsize=(10, 6))
    plt.plot(a_discounts, a_avg_times, marker='o', label='Policy Iteration', color='b')
    plt.plot(b_discounts, b_avg_times, marker='x', label='Value Iteration', color='r')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Discount Factor')
    plt.ylabel('Average Time')
    plt.title('Average Runtime vs Discount Factor')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig('plot.png')


if __name__ == "__main__":
    main()
