import numpy as np

class Agent:

    def __init__(self, env, epsilon, min_epsilon, decay_rate, learning_rate, discount_factor):
        self.env = env
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.Q = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        self.unsafeProb = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        self.failureCount = np.zeros((self.env.observation_space.n, self.env.action_space.n))

    def choose_action(self, observation):
        action = 0
        if np.random.uniform(0, 1) < self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.Q[observation, :])
        return action

    def learn(self, observation, observation2, reward, action):
        prediction = self.Q[observation, action]
        target = reward + self.discount_factor * np.max(self.Q[observation2, :])
        self.Q[observation, action] = self.Q[observation, action] + self.learning_rate * (target - prediction)

    def learnUnsafe(self, observation, action, reward, threeshold):
        if(reward < threeshold):
            self.unsafeProb[observation, action] = 1

    def learnFailureCountWithMax(self, observation, observation2, action):
        prediction = self.failureCount[observation, action]
        target = self.unsafeProb[observation2, action] + 0.99 * self.failureCount[observation2, np.argmax(self.Q[observation2, :])]
        self.failureCount[observation, action] = prediction + 0.1 * (target - prediction)

    def learnFailureCountWithAvg(self, observation, observation2, action):
        prediction = self.failureCount[observation, action]
        target = self.unsafeProb[observation2, action] + 0.99 * np.average(self.failureCount[observation2, :])
        self.failureCount[observation, action] = prediction + 0.1 * (target - prediction)