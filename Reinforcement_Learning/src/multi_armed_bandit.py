import numpy as np


class MultiArmedBandit:
    """
    MultiArmedBandit reinforcement learning agent.

    Arguments:
      epsilon - (float) The probability of randomly exploring the action space
        rather than exploiting the best action.
    """

    def __init__(self, epsilon=0.2):
        self.epsilon = epsilon

    def fit(self, env, steps=1000):
        """
        Trains the MultiArmedBandit on an OpenAI Gym environment.

        See page 32 of Sutton and Barto's book Reinformcement Learning for
        pseudocode (http://incompleteideas.net/book/RLbook2018.pdf).
        Initialize your parameters as all zeros. For the step size (alpha), use
        1 / N, where N is the number of times the current action has been
        performed. Use an epsilon-greedy policy for action selection.

        See (`https://gym.openai.com`/) for examples of how to use the OpenAI
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
          - MultiArmedBandit treats all environment states the same. However,
            in order to have the same API as agents that model state, you must
            explicitly return the state-action-values Q(s, a). To do so, just
            copy the action values learned by MultiArmedBandit S times, where
            S is the number of states.

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
            (Episode = 100)?
        """
        S = env.observation_space.n
        A = env.action_space.n
        state_action_values = np.zeros((S,A))       #each action's number is also the index of the action. This is an 1*m matrix
        action_counts = np.zeros((S,A))                 #number of executions of each action.
        epsilon_count = 0       #counting epsilon
        epsilon_count_bound = np.floor(1.0/self.epsilon)      #

        rewards= np.zeros(100)
        reward_index = 0      #upper bound is np.floor(steps / 100)
        count_rewards = 1       #for calculating the average in rewards

        observation = env.reset()
        for n in range(0, steps):          #this is one huge episode. n is [1...steps]
            env.render()

            if epsilon_count != epsilon_count_bound:
                if np.array_equal(state_action_values[observation,:], np.zeros(A)):
                    action = np.random.randint(0,A)
                    # print("all zero actions!")
                else:
                    action = np.argmax(state_action_values[observation,:])            #assume observation is updated and it is the same as the row number
                epsilon_count += 1
            else:
                action = np.random.randint(0,A)
                epsilon_count = 0

            next_observation, reward, done, info = env.step(action)             #TODO: don't forget to update observation

            #update state_action_values
            action_counts[observation, action] += 1
            state_action_values[observation,action] += 1.0/action_counts[observation, action] * (reward - state_action_values[observation, action])       #TODO

            #update rewards
            if count_rewards < np.floor(steps / 100.0):     #the same s steps
                rewards[reward_index] += 1.0/count_rewards * ( reward - rewards[reward_index]  )
                count_rewards+=1
            else:       #starting a new sequence of s steps
                reward_index += 1
                count_rewards = 1
                if reward_index < 100:
                    rewards[reward_index] += 1.0/count_rewards * ( reward - rewards[reward_index]  )

            if done == True:    #not game over
                # print("multiarmed state action: ", state_action_values)
                observation = env.reset()

        env.close()
        return state_action_values, rewards


    def predict(self, env, state_action_values):
        """
        Runs prediction on an OpenAI environment using the policy defined by
        the MultiArmedBandit algorithm and the state action values. Predictions
        are run for exactly one episode. Note that one episode may produce a
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

        states = np.array([env.reset()])        #assume we're dealing with one state here
        actions = np.array([ np.argmax(state_action_values[0]) ])

        action = actions[0]
        rewards = np.array([ env.step(action) ])
        return states, actions, rewards
