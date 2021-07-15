import numpy
import gym
from qlearnagent import Agent
import csv
import pickle

total_episodes = 50000
max_steps = 300

env = gym.make('FrozenLake-v0')
env.seed(1)
print(env.render())
qlearn = Agent(env, 0.99, 0.01, 0.9, 0.1, 0.96)

total_success = 0
total_failures = 0

with open('frozenlake5.csv', 'w', newline='', encoding='utf-8') as f:
  fieldnames = ['number_of_steps', 'episode_number', 'won', 'algorithm', 'state', 'Q_value_state']

  writer = csv.DictWriter(f, fieldnames=fieldnames)

  writer.writeheader()

  for episode in range(total_episodes):
    obs = env.reset()
    env.seed(1)
    t = 0
    if episode % 100 == 99:
      qlearn.epsilon *= qlearn.decay_rate
      qlearn.epsilon = max(qlearn.epsilon, qlearn.min_epsilon)
    while t < max_steps:
      action = qlearn.choose_action(obs)
      obs2, reward, done, info = env.step(action)
      # print('observation', obs2)
      # print('action', action)
      # unsafe = 0
      # qlearn.learnFailureCountWithMax(obs, obs2, action, unsafe)

      # if(qlearn.failureCount[obs, action] > numpy.average(qlearn.failureCount[:,:])):
      #   qlearn.discount_factor = 0.96
      # else:
      #   qlearn.discount_factor = 0.4

      qlearn.learn(obs, obs2, reward, action)
      t += 1
      # SFFF
      # FHFH
      # FFFH
      # HFFG

      if done:
        if reward > 0.0:
              writer.writerow({'number_of_steps': t, 'episode_number': episode, 'won': 1, 'algorithm': 'standard'})
              total_success = total_success + 1
        else:
              if(obs2 == 5 and action == 2):
                writer.writerow({'number_of_steps': t, 'episode_number': episode, 'won': 0, 'algorithm': 'standard',
                  'state': obs2, 'Q_value_state': qlearn.Q[obs, action]})
              else:
                writer.writerow({'number_of_steps': t, 'episode_number': episode, 'won': 0, 'algorithm': 'standard'})
              # unsafe = 1
              # qlearn.learnFailureCountWithMax(obs, obs2, action, unsafe)
              # if(qlearn.failureCount[obs, action] > numpy.average(qlearn.failureCount[:,:])):
              #   qlearn.discount_factor = 0.96
              # else:
              #   qlearn.discount_factor = 0.4

              # qlearn.learn(obs, obs2, reward, action)
              total_failures = total_failures + 1
        break
      else:
        obs = obs2



print("total success dynamic discount factor", total_success)
print("total failures discount factor", total_failures)

print(qlearn.failureCount)
print(qlearn.Q)

env.close()