import numpy
import gym
from qlearnagent import Agent

total_episodes = 50000
max_steps = 300

env = gym.make('FrozenLake-v0')
env.seed(1)

print(env.render())
# start with smaller discount factor
qlearn = Agent(env, 0.99, 0.01, 0.9, 0.1, 0.1)

total_success = 0
total_failures = 0

for episode in range(total_episodes):
  obs = env.reset()
  env.seed(1)
  rewardsList = []
  t = 0
  if episode % 100 == 99:
    qlearn.epsilon *= qlearn.decay_rate
    qlearn.epsilon = max(qlearn.epsilon, qlearn.min_epsilon)
  while t < max_steps:
    action = qlearn.choose_action(obs)
    obs2, reward, done, info = env.step(action)
    qlearn.learn(obs, obs2, reward, action)
    obs = obs2

    qlearn.R[obs2, action] = reward
    rewardsList.append({obs2 : action})

    t += 1
    if done:
      if reward > 0.0:
            total_success = total_success + 1
      else:
            qlearn.R[obs2, action] = -1
            if(len(rewardsList) == 3):
              qlearn.calculateProbError(rewardsList)
            total_failures = total_failures + 1
      break
    else:
      if(len(rewardsList) == 3):
        qlearn.calculateProbError(rewardsList)
        # rewardsList.pop(0)

simpleQlearn = Agent(env, 0.99, 0.01, 0.9, 0.1, 0.96)

total_success_simple = 0
total_failures_simple = 0

for episode in range(total_episodes):
  obs = env.reset()
  env.seed(1)
  t = 0
  if episode % 100 == 99:
    simpleQlearn.epsilon *= simpleQlearn.decay_rate
    simpleQlearn.epsilon = max(simpleQlearn.epsilon, simpleQlearn.min_epsilon)
  while t < max_steps:
    action = simpleQlearn.choose_action(obs)
    obs2, reward, done, info = env.step(action)
    simpleQlearn.learn(obs, obs2, reward, action)
    obs = obs2

    t += 1
    if done:
      if reward > 0.0:
            total_success_simple = total_success_simple + 1
      else:
            total_failures_simple = total_failures_simple + 1
      break
        # rewardsList.pop(0)
print("total success dynamic discount factor", total_success)
print("total failures discount factor", total_failures)
print("total success", total_success_simple)
print("total failures", total_failures_simple)

print(qlearn.R)
print(qlearn.FailProb)
print(qlearn.Q)
print(simpleQlearn.Q)
env.close()