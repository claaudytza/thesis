import numpy
import gym
from qlearnagent import Agent
import csv

total_episodes = 50000
max_steps = 1000

env = gym.make('gym_waterworld:waterworld-v0')

print(env.render())

# start with smaller discount factor
qlearn = Agent(env, 0.99, 0.01, 0.9, 0.1, 0.1)

total_success = 0
total_failures = 0

for episode in range(total_episodes):
  obs = env.reset()

  t = 0
  if episode % 100 == 99:
    qlearn.epsilon *= qlearn.decay_rate
    qlearn.epsilon = max(qlearn.epsilon, qlearn.min_epsilon)
  while t < max_steps:
    action = qlearn.choose_action(obs)
    obs2, reward, done, info = env.step(action)

    qlearn.learnUnsafe(obs2, action, reward, 0)
    qlearn.learnFailureCountWithAvg(obs, obs2, action)

    if(qlearn.failureCount[obs2, action] > 10):
      qlearn.discount_factor = 0.96
    else:
      qlearn.discount_factor = 0.1

    qlearn.learn(obs, obs2, reward, action)
    obs = obs2

    t += 1
    if done:
      if reward > 0.0:
            total_success = total_success + 1
      else:
            total_failures = total_failures + 1
      break


print("total success dynamic discount factor", total_success)
print("total failures discount factor", total_failures)

print(qlearn.failureCount)
print(qlearn.unsafeProb)
print(qlearn.Q)

env.close()