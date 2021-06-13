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
        self.R = numpy.zeros((self.env.observation_space.n, self.env.action_space.n))
        self.Fail = numpy.zeros((self.env.observation_space.n, self.env.action_space.n))

    def choose_action(self, observation):
        action = 0
        if numpy.random.uniform(0, 1) < self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = numpy.argmax(self.Q[observation, :])
        return action

    def learn(self, observation, observation2, reward, action):
        prediction = self.Q[observation, action]
        if(self.Fail[observation, action] != 0):
            calculated_discount = self.Fail[observation, action]
            target = reward + calculated_discount * numpy.max(self.Q[observation2, :])
        else:
            target = reward + self.discount_factor * numpy.max(self.Q[observation2, :])
        self.Q[observation, action] = self.Q[observation, action] + self.learning_rate * (target - prediction)

    def calculateDiscount(self, observations):
        failures = 0
        for observation in observations:
            for key in observation.keys():
                if(self.R[key, observation[key]] < 0):
                    failures += 1

        if(failures > 0):
            prob_fail = failures/len(observations)
            first_obs = observations[0]
            for key in first_obs.keys():
                self.Fail[key, first_obs[key]] = prob_fail
