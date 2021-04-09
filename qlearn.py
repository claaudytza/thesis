import numpy
import gym

#epsilon-greedy approach
epsilon = 0.99
min_epsilon = 0.01
decay_rate = 0.9
total_episodes = 50000
max_steps = 100
learning_rate = 0.1
#discount factor
gamma = 0.96

env = gym.make('FrozenLake-v0')

#initialize matrix
Q = numpy.zeros((env.observation_space.n, env.action_space.n))

times_success = []

total_success = 0
total_failures = 0

#epsilon-greedy approach - randomly generate a number between 0 and 1 and 
# see if it is smaller than epsilon > yes -> take a random action, > not then choose 
#the action having the maximum value in the Q-table for the state - ensure exploration and exploitation
def choose_action(observation):
  action = 0
  if numpy.random.uniform(0, 1) < epsilon:
    action = env.action_space.sample()
  else:
    action = numpy.argmax(Q[observation, :])
  return action


#update the table - using the equation 
def learn(observation, observation2, reward, action):
  prediction = Q[observation, action]
  target = reward + gamma * numpy.max(Q[observation2, :])
  Q[observation, action] = Q[observation, action] + learning_rate * (target - prediction)


for episode in range(total_episodes):
  obs = env.reset()
  t = 0
  if episode % 100 == 99:
    epsilon *= decay_rate
    epsilon = max(epsilon, min_epsilon)
  while t < max_steps:
    action = choose_action(obs)
    obs2, reward, done, info = env.step(action)
    learn(obs, obs2, reward, action)
    obs = obs2
    t += 1
    if done:
      if reward > 0.0:
            total_success = total_success + 1
            times_success.append(episode)
      else:
            total_failures = total_failures + 1
      break

# print(times_success)
print("total success", total_success)
print("total failures", total_failures)

print(Q)

env.close()