import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

env_4x4 = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
env_8x8 = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True)

# Parameters

num_episodes = 20000
alpha = 0.3
gamma = 0.9
epsilon = 0.6

# def td_zero(env, num_episodes, alpha=0.1, gamma=0.9):
#     V = np.zeros(env.observation_space.n)
#     mean_values = []
#
#     for episode in range(num_episodes):
#         state, _ = env.reset()
#         done = False
#         while not done:
#             action = env.action_space.sample()
#             next_state, reward, done, _, _ = env.step(action)
#             V[state] += alpha * (reward + gamma * V[next_state] - V[state])
#             state = next_state
#         mean_values.append(np.mean(V))
#
#     return V, mean_values



# TD(0) Algorithm (Epsilon Greedy)

def td_zero(env, episodes, alpha, gamma, epsilon):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    V = np.zeros(n_states)  # Initialize value function
    mean_values = []

    for _ in range(episodes):
        state, _ = env.reset()
        done = False

        while not done:
            # ε-greedy action selection
            if np.random.rand() < epsilon:
                action = np.random.choice(n_actions)
            else:
                action = np.argmax([V[state] for _ in range(n_actions)])

            next_state, reward, done, _, _ = env.step(action)
            V[state] += alpha * (reward + gamma * V[next_state] - V[state])
            state = next_state

        mean_values.append(np.mean(V))

    return V, mean_values

# def monte_carlo(env, num_episodes, alpha=0.1, gamma=0.9):
#     V = np.zeros(env.observation_space.n)
#     returns = {s: [] for s in range(env.observation_space.n)}
#     mean_values = []
#
#     for episode in range(num_episodes):
#         episode_data = []
#         state, _ = env.reset()
#         done = False
#         while not done:
#             action = env.action_space.sample()
#             next_state, reward, done, _, _ = env.step(action)
#             episode_data.append((state, reward))
#             state = next_state
#
#         G = 0
#         for state, reward in reversed(episode_data):
#             G = reward + gamma * G
#             if state not in [x[0] for x in episode_data[:episode_data.index((state, reward))]]:
#                 returns[state].append(G)
#                 V[state] += alpha * (G - V[state])
#
#         mean_values.append(np.mean(V))
#
#     return V, mean_values


# Monte Carlo (MC) Algorithm (Epsilon Greedy)

def monte_carlo(env, episodes, alpha, gamma, epsilon):
    n_states = env.observation_space.n
    V = np.zeros(n_states)
    returns = {s: [] for s in range(n_states)}
    mean_values = []

    for _ in range(episodes):
        state, _ = env.reset()
        episode = []
        done = False

        while not done:
            # ε-greedy action selection
            if np.random.rand() < epsilon:
                action = np.random.choice(env.action_space.n)
            else:
                action = np.argmax([V[state] for _ in range(env.action_space.n)])

            next_state, reward, done, _, _ = env.step(action)
            episode.append((state, reward))
            state = next_state

        G = 0
        for t in reversed(range(len(episode))):
            state, reward = episode[t]
            G = gamma * G + reward
            if state not in [x[0] for x in episode[:t]]:  # First-visit MC
                returns[state].append(G)
                V[state] += alpha * (G - V[state])

        mean_values.append(np.mean(V))

    return V, mean_values



def plot_convergence(env, num_episodes, alpha, gamma, epsilon, size):


    # TD(0) Convergence
    V_TD, td_mean_values = td_zero(env, num_episodes, alpha, gamma, epsilon)

    # print(V_TD)

    # MC Convergence
    V_MC, mc_mean_values = monte_carlo(env, num_episodes, alpha, gamma, epsilon)

    #print(V_MC)

    # Plot Value Function Convergence over Episodes
    plt.figure(figsize=(10, 5))
    plt.plot(range(num_episodes), td_mean_values, label="TD(0)", color="blue")
    plt.plot(range(num_episodes), mc_mean_values, label="MC", color="green")
    plt.xlabel("Episodes")
    plt.ylabel("Mean Value of V")
    plt.title(f"Value Function Convergence Over Episodes for {size}x{size} Frozen Lake")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot TD(0) and MC Value Functions
    plt.figure(figsize=(12, 6))

    # TD(0) values
    plt.subplot(1, 2, 1)
    plt.title(f"TD(0) Value Function for {size}x{size} Frozen Lake")
    plt.imshow(V_TD.reshape(size, size), cmap="coolwarm", interpolation="nearest")
    plt.colorbar()

    # MC values
    plt.subplot(1, 2, 2)
    plt.title(f"MC Value Function for {size}x{size} Frozen Lake")
    plt.imshow(V_MC.reshape(size, size), cmap="coolwarm", interpolation="nearest")
    plt.colorbar()

    plt.show()


# Run
# For 4x4 Frozen Lake
plot_convergence(env_4x4, num_episodes, alpha, gamma, epsilon, size=4)

# For 8x8 Frozen Lake
plot_convergence(env_8x8, num_episodes, alpha, gamma, epsilon, size=8)