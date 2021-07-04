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
        self.FailProb = numpy.zeros((self.env.observation_space.n, self.env.action_space.n))

    def choose_action(self, observation):
        action = 0
        if numpy.random.uniform(0, 1) < self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = numpy.argmax(self.Q[observation, :])
        return action

    def learn(self, observation, observation2, reward, action):
        prediction = self.Q[observation, action]
        # target = 0
        if(self.FailProb[observation, action] != 0):
            if(self.FailProb[observation, action] >= 0.5):
                target = reward + 0.96 * numpy.max(self.Q[observation2, :])
            else:
                target = reward + 0.5 * numpy.max(self.Q[observation2, :])
            # print("first if ", 0.96 * 0)
        else:
            target = reward + self.discount_factor * numpy.max(self.Q[observation2, :])
            # print("second if", target)
        # print(target)
        self.Q[observation, action] = self.Q[observation, action] + self.learning_rate * (target - prediction)

    def calculateProbError(self, observations):
        total_count = 0
        failures = 0

        first_obs = observations.pop(0)

        for observation in observations:
            for key in observation.keys():
                for obs in self.R[key, :]:
                    total_count += 1
                    if(obs < 0):
                        failures += 1

        if(failures > 0):
            # print(failures, end="\n")
            # print(total_count, end="\n")
            prob_fail = failures/total_count
            for key in first_obs.keys():
                self.FailProb[key, first_obs[key]] = prob_fail

# make 0.1, 0.5 and 0.9 discount factor, based on if it is 0, inbetween 0 and 5, and 5 and 1
# give the weights based on penalties and make sure to round probability error
# grass field environment