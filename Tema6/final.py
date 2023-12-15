import numpy as np
import matplotlib.pyplot as plt
from colorama import Fore, Style

number_of_rows = 7
number_of_columns = 10
initial_state = (3, 0)
goal_state = (3, 7)
wind_strength = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

Q_table = np.zeros((number_of_rows, number_of_columns, 4))
alpha = 0.2  # learning rate
gamma = 0.6  # discount factor
epsilon = 0.3
number_of_episodes = 100

actions = ["↑", "↓", "←", "→"]


def get_next_state(current_state, action):
    row = current_state[0]
    column = current_state[1]
    wind = wind_strength[column]

    if action == 0:  # up
        row -= 1
    elif action == 1:  # down
        row += 1
    elif action == 2:  # left
        column -= 1
    elif action == 3:  # right
        column += 1

    row -= wind

    row = max(0, min(row, number_of_rows - 1))
    column = max(0, min(column, number_of_columns - 1))

    return row, column


episode_rewards = []

for episode in range(number_of_episodes):
    current_state = initial_state
    total_reward = 0

    while current_state != goal_state:
        action = np.argmax(Q_table[current_state[0], current_state[1]])
        next_state = get_next_state(current_state, action)
        if next_state != goal_state:
            reward = -1
        else:
            reward = 1000

        best_next_action = np.argmax(Q_table[next_state[0], next_state[1]])

        Q_table[current_state[0], current_state[1], action] += alpha * (
                reward + gamma * Q_table[next_state[0], next_state[1], best_next_action] -
                Q_table[current_state[0], current_state[1], action]
        )

        current_state = next_state
        total_reward += reward

    episode_rewards.append(total_reward)


def get_best_action(Q_table, current_state):
    return np.argmax(Q_table[current_state[0], current_state[1]])


current_state = initial_state
path = [current_state]

while current_state != goal_state:
    best_action = get_best_action(Q_table, current_state)
    next_state = get_next_state(current_state, best_action)
    current_state = next_state
    path.append(current_state)

print("Optimal Path:")
for state in path:
    print(state)

policy = np.argmax(Q_table, axis=2)
policy_with_arrows = np.array([["G" if (i, j) == goal_state else "S" if (i,j) == initial_state else actions[action] for j, action in enumerate(row)] for i, row in enumerate(policy)])

print("Policy:")
print(policy_with_arrows)
print(wind_strength)


policy = np.argmax(Q_table, axis=2)
policy_with_arrows = np.array([["G" if (i, j) == goal_state else "S" if (i,j) == initial_state else "x" if (i,j) not in path else actions[action] for j, action in enumerate(row)] for i, row in enumerate(policy)])

print("Policy:")
print(policy_with_arrows)

plt.plot(episode_rewards)
plt.title('Rewards Over Time')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()