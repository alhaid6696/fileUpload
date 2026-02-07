
# =========================================================
# Q-Learning Implementation for CartPole-v1
# Course: COEN 874 - Reinforcement Learning
# =========================================================

import gym
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Environment Setup
# -------------------------------
env = gym.make("CartPole-v1")

# -------------------------------
# Discretization Parameters
# -------------------------------
NUM_BINS = 10

state_bounds = [
    (-4.8, 4.8),       # Cart position
    (-4.0, 4.0),       # Cart velocity
    (-0.418, 0.418),   # Pole angle
    (-4.0, 4.0)        # Pole angular velocity
]

bins = [
    np.linspace(state_bounds[i][0], state_bounds[i][1], NUM_BINS)
    for i in range(4)
]

# -------------------------------
# Q-Table Initialization
# -------------------------------
q_table = np.zeros((NUM_BINS, NUM_BINS, NUM_BINS, NUM_BINS, env.action_space.n))

# -------------------------------
# Hyperparameters
# -------------------------------
alpha = 0.1          # Learning rate
gamma = 0.99         # Discount factor
epsilon = 1.0        # Initial exploration rate
epsilon_decay = 0.995
epsilon_min = 0.01
episodes = 500

# -------------------------------
# Helper Function
# -------------------------------
def discretize_state(state):
    indices = []
    for i in range(len(state)):
        index = np.digitize(state[i], bins[i]) - 1
        index = max(0, min(NUM_BINS - 1, index))
        indices.append(index)
    return tuple(indices)

# -------------------------------
# Training Loop
# -------------------------------
episode_rewards = []

for episode in range(episodes):
    state, _ = env.reset()
    state = discretize_state(state)
    total_reward = 0
    done = False

    while not done:
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, done, truncated, _ = env.step(action)
        next_state = discretize_state(next_state)

        # Q-learning update
        q_table[state][action] += alpha * (
            reward + gamma * np.max(q_table[next_state]) - q_table[state][action]
        )

        state = next_state
        total_reward += reward

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    episode_rewards.append(total_reward)

    if (episode + 1) % 50 == 0:
        print(f"Episode {episode + 1}, Average Reward: {np.mean(episode_rewards[-50:]):.2f}")

# -------------------------------
# Plot Results
# -------------------------------
plt.figure()
plt.plot(episode_rewards)
plt.xlabel("Episodes")
plt.ylabel("Cumulative Reward")
plt.title("Q-Learning Performance on CartPole-v1")
plt.show()

env.close()
