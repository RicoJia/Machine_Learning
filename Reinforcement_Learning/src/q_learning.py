import numpy as np


class QLearning:
    """
    QLearning reinforcement learning agent.

    Arguments:
      epsilon - (float) The probability of randomly exploring the action space
        rather than exploiting the best action.
      discount - (float) The discount factor. Controls the perceived value of
        future reward relative to short-term reward.
      adaptive - (bool) Whether to use an adaptive policy for setting
        values of epsilon during training
    """

    def __init__(self, epsilon=0.5, discount=0.95, adaptive=False):
        self.epsilon = epsilon
        self.discount = discount
        self.adaptive = adaptive

    def fit(self, env, steps=1000):
        """
        Trains an agent using Q-Learning on an OpenAI Gym Environment.

        See page 131 of Sutton and Barto's book Reinformcement Learning for
        pseudocode (http://incompleteideas.net/book/RLbook2018.pdf).
        Initialize your parameters as all zeros. For the step size (alpha), use
        1 / N, where N is the number of times the current action has been
        performed in the current state. Note that this is a different formula
        for the step size than was used in MultiArmedBandits. Use an
        epsilon-greedy policy for action selection. Note that unlike the
        pseudocode, we are looping over a total number of steps, and not a
        total number of episodes. This allows us to ensure that all of our
        trials have the same number of steps--and thus roughly the same amount
        of computation time.

        See (https://gym.openai.com/) for examples of how to use the OpenAI
        Gym Environment interface.

        Hints:
          - Use env.action_space.n and env.observation_space.n to get the
            number of available actions and states, respectively.
          - Remember to reset your environment at the end of each episode. To
            do this, call env.reset() whenever the value of "done" returned
            from env.step() is True.
          - If all values of a np.array are equal, np.argmax deterministically
            returns 0.
          - In order to avoid non-deterministic tests, use only np.random for
            random number generation.
          - Use the provided self._get_epsilon function whenever you need to
            obtain the current value of epsilon.

        Arguments:
          env - (Env) An OpenAI Gym environment with discrete actions and
            observations. See the OpenAI Gym documentation for example use
            cases (https://gym.openai.com/docs/).
          steps - (int) The number of actions to perform within the environment
            during training.

        Returns:
          state_action_values - (np.array) The values assigned by the algorithm
            to each state-action pair as a 2D numpy array. The dimensionality
            of the numpy array should be S x A, where S is the number of
            states in the environment and A is the number of possible actions.
          rewards - (np.array) A 1D sequence of averaged rewards of length 100.
            Let s = np.floor(steps / 100), then rewards[0] should contain the
            average reward over the first s steps, rewards[1] should contain
            the average reward over the next s steps, etc.
        """
        S = env.observation_space.n
        A = env.action_space.n
        state_action_values = np.zeros(
            (S, A)
        )  # each action's number is also the index of the action. This is an S x A matrix
        action_counts = np.zeros(
            (S, A)
        )  # number of executions of each action. S x A matrux

        epsilon_count = 0  # counting epsilon, so every time it hits the bound, it will draw a random action, (not using argmax!)
        epsilon_count_bound = np.floor(1.0 / self.epsilon)  # this might be adaptive

        rewards = np.zeros(100)  # average rewards for every s steps.
        reward_index = 0  # upper bound is np.floor(steps / 100)
        count_rewards = 1  # for calculating the average in rewards

        observation = (
            env.reset()
        )  # TODO: not sure if we should reset the environment just once, since we are not doing episodes?

        for n in range(0, steps):  # this is one huge episode. n is [1...steps]
            env.render()

            # get action
            if self.adaptive == True:
                _epsilon = self._adaptive_epsilon(float(n) / steps)
                epsilon_count_bound = np.floor(1.0 / _epsilon)

            if epsilon_count != epsilon_count_bound:
                if np.array_equal(state_action_values[observation, :], np.zeros(A)):
                    action = np.random.randint(0, A)
                    # print("all zero actions!")
                else:
                    action = np.argmax(
                        state_action_values[observation, :]
                    )  # assume observation is updated and it is the same as the row number
                epsilon_count += 1
            else:
                action = np.random.randint(0, A - 1)
                epsilon_count = 0
                # print("randomized!")

            next_observation, reward, done, info = env.step(
                action
            )  # TODO: don't forget to update observation

            # update state_action_values
            action_counts[observation, action] += 1
            state_action_values[observation, action] += (
                1.0
                / action_counts[observation, action]
                * (
                    reward
                    + self.discount * np.max(state_action_values[next_observation, :])
                    - state_action_values[observation, action]
                )
            )  # TODO

            # update rewards
            if count_rewards < np.floor(steps / 100.0):  # the same s steps
                rewards[reward_index] += (
                    1.0 / count_rewards * (reward - rewards[reward_index])
                )
                count_rewards += 1
            else:  # starting a new sequence of s steps
                reward_index += 1
                count_rewards = 1
                if reward_index < 100:
                    rewards[reward_index] += (
                        1.0 / count_rewards * (reward - rewards[reward_index])
                    )

            observation = next_observation

            if done == True:  # not game over
                observation = env.reset()

        env.close()

        return state_action_values, rewards

    def predict(self, env, state_action_values):
        """
        Runs prediction on an OpenAI environment using the policy defined by
        the QLearning algorithm and the state action values. Predictions are
        run for exactly one episode. Note that one episode may produce a
        variable number of steps.

        Hints:
          - You should not update the state_action_values during prediction.
          - Exploration is only used in training. Any mechanisms used for
            exploration in the training phase should not be used in prediction.

        Arguments:
          env - (Env) An OpenAI Gym environment with discrete actions and
            observations. See the OpenAI Gym documentation for example use
            cases (https://gym.openai.com/docs/).
          state_action_values - (np.array) The values assigned by the algorithm
            to each state-action pair as a 2D numpy array. The dimensionality
            of the numpy array should be S x A, where S is the number of
            states in the environment and A is the number of possible actions.

        Returns:
          states - (np.array) The sequence of states visited by the agent over
            the course of the episode. Does not include the starting state.
            Should be of length K, where K is the number of steps taken within
            the episode.
          actions - (np.array) The sequence of actions taken by the agent over
            the course of the episode. Should be of length K, where K is the
            number of steps taken within the episode.
          rewards - (np.array) The sequence of rewards received by the agent
            over the course  of the episode. Should be of length K, where K is
            the number of steps taken within the episode.
        """
        states = np.array([])
        state = env.reset()

        actions = np.array([])
        rewards = np.array([])
        done = False
        while done == False:  # not at the terminal yet
            env.render()
            action = np.argmax(state_action_values[state])
            state, reward, done, info = env.step(action)
            states = np.append(states, state)
            actions = np.append(actions, action)
            rewards = np.append(rewards, reward)

        return states, actions, rewards

    def _get_epsilon(self, progress):
        """
        Retrieves the current value of epsilon. Should be called by the fit
        function during each step.

        Arguments:
            progress - (float) A value between 0 and 1 that indicates the
                training progess. Equivalent to current_step / steps.
        """
        return self._adaptive_epsilon(progress) if self.adaptive else self.epsilon

    def _adaptive_epsilon(self, progress):
        """
        An adaptive policy for epsilon-greedy reinforcement learning. Returns
        the current epsilon value given the learner's progress. This allows for
        the amount of exploratory vs exploitatory behavior to change over time.

        See free response question 3 for instructions on how to implement this
        function.

        Arguments:
            progress - (float) A value between 0 and 1 that indicates the
                training progess. Equivalent to current_step / steps.
        """
        return (1.0 - progress) * self.epsilon
