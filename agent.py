import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6, alpha=0.02, gamma=0.9):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.alpha = alpha
        self.gamma = gamma
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.policy = {}

    def select_action(self, state, probs):
        """ Given the state, select an action. Also update the agent policy.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        self.policy[state] = np.random.choice(np.arange(self.nA), p=probs)
        action = self.policy[state]

        return action

    def calculate_epsilon(self, i_episode, num_episodes):
    	""" Calculate the epsilon for the epsilon greedy policy
    	
    	Params
    	======
		- i_episode: the current episode
		- num_episodes: the number of episodes

    	Returns
    	=======
    	- epsilon: the parameter used for epsilon greedy policy
    	"""

    	epsilon = 1/i_episode

    	return epsilon


    def epsilon_greedy_policy_probs(self, state, epsilon):
    	""" Calculate the policy probs for epsilon greedy policy
		
		Params
		======
		- state: the current state the
		- epsilon: the current calculated value of epsilon

		Returns
		=======
		- probs: the updated mapping from state to action
    	"""

    	probs = np.ones(self.nA)*(epsilon/self.nA)
    	probs[np.argmax(self.Q[state])] = 1 - epsilon + epsilon/self.nA

    	return probs


    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        if not done:
        	self.Q[state][action] += self.alpha*(reward + self.gamma*(max(self.Q[next_state])) - self.Q[state][action])
        else:
        	self.Q[state][action] += self.alpha*(reward - self.Q[state][action])