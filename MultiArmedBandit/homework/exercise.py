import matplotlib.pyplot as plt
import numpy as np

class Bandit:
    def __init__(self, k):
        self.k = k
        self.means = np.random.normal(0, 3, k)
        self.arms = [self._pull_arm(mean) for mean in self.means]

    def _pull_arm(self, mean):
        return lambda: np.random.normal(mean, 1)

    def pull(self, arm):
        return self.arms[arm]()


def run_agent(bandit, algorithm, steps):
    rewards = np.zeros(steps)
    for step in range(steps):
        arm = algorithm.selector()
        reward = bandit.pull(arm)
        algorithm.update(arm, reward)
        rewards[step] = reward
    return rewards


class agent:
    def __init__(self, k):
        self.k = k
        self.qvals = np.zeros(k)
        self.arm_counts = np.zeros(k)

    def update(self, arm, reward):
        self.arm_counts[arm] += 1
        # n+1
        self.qvals[arm] += (reward - self.qvals[arm]) / self.arm_counts[arm]


class EGreedy(agent):
    def __init__(self, k, eps):
        super().__init__(k)
        self.eps = eps

    def selector(self):
        if np.random.random() > self.eps:
            return np.random.randint(0, self.k)
        else:
            return np.argmax(self.qvals)


class OptisticGreed(agent):
    def __init__(self, k, scale):
        super().__init__(k)
        self.qvals = np.ones(self.k) * scale

    def selector(self):
        return np.argmax(self.qvals)


class UCB(agent):
    def __init__(self, k, tradeoff):
        super().__init__(k)
        self.tradeoff = tradeoff
        # avoid log zero issues
        self.total = 1

    def selector(self):
        self.total += 1
        # avoid div by 0 issues
        offset = 0.1
        return np.argmax(
            self.qvals + self.tradeoff * (np.sqrt(np.divide(np.log(self.total), self.arm_counts + offset))))

k = 10
steps = 1000
runs = 100

epsilon = 0.1
optimism_scale = 5
c = 2

algorithms = [
    (EGreedy(k, epsilon), "E-Greedy"),
    (OptisticGreed(k, optimism_scale), "Optimistic Initial Values (Greedy)"),
    (UCB(k, c), "UCB")
]

plt.figure(figsize=(12, 8))

for algorithm, label in algorithms:
    all_rewards = np.zeros(steps)
    for _ in range(runs):
        bandit = Bandit(k)
        rewards = run_agent(bandit, algorithm, steps)
        all_rewards += rewards
    all_rewards /= runs
    plt.plot(all_rewards, label=label)

plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.title("Average Reward over Time for Different Algorithms")
plt.legend()
plt.show()

# plt.savefig("exercise.png")


