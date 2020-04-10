import gym
import numpy as np
from src import MultiArmedBandit
import matplotlib.pyplot as plt
from src import QLearning


def FRQ_2_a():
    env = gym.make('SlotMachines-v0')
    rewards_ls = []
    for i in range(10):
        agent = MultiArmedBandit()
        action_values, rewards = agent.fit(env, steps=100000)
        rewards_ls.append(rewards)


    #plot the rewards of the first agent
    plt.plot(np.linspace(0,99,100),rewards_ls[0])

    sum_5_trials = np.zeros(100)
    for i in range(5):
        sum_5_trials += rewards_ls[i]
    average_5_trials = 1.0/5.0 * sum_5_trials
    plt.plot(np.linspace(0,99,100), average_5_trials)

    sum_10_trials = np.zeros(100)
    for i in range(10):
        sum_10_trials = sum_10_trials + rewards_ls[i]
    average_10_trials = 1.0/10.0 * sum_10_trials
    plt.plot(np.linspace(0,99,100), average_10_trials)

    plt.legend(('first trial,', 'first 5 trials', 'all 10 trials'), loc='lower right')
    plt.title("FRQ_2a_Rewards_Comparison")
    plt.xlabel("Steps")
    plt.ylabel("Reward Values")
    plt.show()


def FRQ_2_e():

    env = gym.make('FrozenLake-v0')
    multiarmed_rewards_ls = []
    for i in range(10):
        agent = MultiArmedBandit()
        action_values, multiarmed_rewards = agent.fit(env, steps=100000)
        multiarmed_rewards_ls.append(multiarmed_rewards)

    sum_10_trials = np.zeros(100)
    for i in range(10):
        sum_10_trials = sum_10_trials + multiarmed_rewards_ls[i]
    average_10_trials = 1.0/10.0 * sum_10_trials
    plt.plot(np.linspace(0,99,100), average_10_trials)

    Qlearning_rewards_ls = []
    for i in range(10):
        agent = QLearning()
        action_values, Qlearning_rewards = agent.fit(env, steps=1000)
        Qlearning_rewards_ls.append(Qlearning_rewards)
    sum_10_trials = np.zeros(100)
    for i in range(10):
        sum_10_trials = sum_10_trials + Qlearning_rewards_ls[i]
    average_10_trials = 1.0/10.0 * sum_10_trials
    plt.plot(np.linspace(0,99,100), average_10_trials)

    plt.legend(('MultiArmedbandit', 'Qlearning'), loc='lower right')
    plt.title("FRQ_2e_Qlearning_Multiarmedbandit_rewards_Comparison")
    plt.xlabel("Steps")
    plt.ylabel("Reward Values")
    plt.show()


def FRQ_3_a():

    env = gym.make('FrozenLake-v0')

    rewards_001_ls = []
    for i in range(10):
        agent = QLearning(epsilon = 0.01)
        action_values, Qlearning_rewards = agent.fit(env, steps=100000)
        rewards_001_ls.append(Qlearning_rewards)
    sum_10_trials = np.zeros(100)
    for i in range(10):
        sum_10_trials = sum_10_trials + rewards_001_ls[i]
    average_10_trials = 1.0/10.0 * sum_10_trials
    plt.plot(np.linspace(0,99,100), average_10_trials)

    rewards_05_ls = []
    for i in range(10):
        agent = QLearning(epsilon = 0.5)
        action_values, Qlearning_rewards = agent.fit(env, steps=100000)
        rewards_05_ls.append(Qlearning_rewards)
    sum_10_trials = np.zeros(100)
    for i in range(10):
        sum_10_trials = sum_10_trials + rewards_05_ls[i]
    average_10_trials = 1.0/10.0 * sum_10_trials
    plt.plot(np.linspace(0,99,100), average_10_trials)


    plt.legend(('epsilon=0.01', 'epsilon=0.5'), loc='lower right')
    plt.title("FRQ_3a_Different_Epsilon_Rewards_Comparison")
    plt.xlabel("Steps")
    plt.ylabel("Reward Values")
    plt.show()


def FRQ_3_c():

    env = gym.make('FrozenLake-v0')

    rewards_001_ls = []
    for i in range(10):
        agent = QLearning(epsilon = 0.01)
        action_values, Qlearning_rewards = agent.fit(env, steps=100000)
        rewards_001_ls.append(Qlearning_rewards)
    sum_10_trials = np.zeros(100)
    for i in range(10):
        sum_10_trials = sum_10_trials + rewards_001_ls[i]
    average_10_trials = 1.0/10.0 * sum_10_trials
    plt.plot(np.linspace(0,99,100), average_10_trials)

    rewards_05_ls = []
    for i in range(10):
        agent = QLearning(epsilon = 0.5)
        action_values, Qlearning_rewards = agent.fit(env, steps=100000)
        rewards_05_ls.append(Qlearning_rewards)
    sum_10_trials = np.zeros(100)
    for i in range(10):
        sum_10_trials = sum_10_trials + rewards_05_ls[i]
    average_10_trials = 1.0/10.0 * sum_10_trials
    plt.plot(np.linspace(0,99,100), average_10_trials)


    rewards_adaptive_ls_ls = []
    for i in range(10):
        agent = QLearning(epsilon = 0.5, adaptive=True)
        action_values, Qlearning_rewards = agent.fit(env, steps=100000)
        rewards_adaptive_ls_ls.append(Qlearning_rewards)
    sum_10_trials = np.zeros(100)
    for i in range(10):
        sum_10_trials = sum_10_trials + rewards_adaptive_ls_ls[i]
    average_10_trials = 1.0/10.0 * sum_10_trials
    plt.plot(np.linspace(0,99,100), average_10_trials)


    plt.legend(('epsilon=0.01', 'epsilon=0.5', 'adaptive epsilon'), loc='lower right')
    plt.title("FRQ_3c_Adaptive_Epsilon_Rewards_Comparison")
    plt.xlabel("Steps")
    plt.ylabel("Reward Values")
    plt.show()


FRQ_3_c()
