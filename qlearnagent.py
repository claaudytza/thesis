import numpy

class Agent:

    def __init__(self, env, epsilon, min_epsilon, decay_rate, learning_rate, discount_factor):
        self.env = env
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.Q = numpy.zeros((self.env.observation_space.n, self.env.action_space.n))

    def choose_action(self, observation):
        action = 0
        if numpy.random.uniform(0, 1) < self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = numpy.argmax(self.Q[observation, :])
        return action

    def learn(self, observation, observation2, reward, action):
        prediction = self.Q[observation, action]
        target = reward + self.discount_factor * numpy.max(self.Q[observation2, :])
        self.Q[observation, action] = self.Q[observation, action] + self.learning_rate * (target - prediction)