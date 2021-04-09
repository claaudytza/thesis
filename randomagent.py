import gym
import numpy as np
import random

env = gym.make('FrozenLake-v0')

times_success = []

total_success = 0
total_failures = 0

for i_episode in range(50000):
    observation = env.reset()
    for t in range(100):
        #By doing sample, we use the gym ai random sampling so we do not need to seed ourselves
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            if(reward != 0):
                total_success = total_success + 1
                times_success.append(i_episode)
            else:
                total_failures = total_failures + 1
            break

# print(times_success)
print("total success", total_success)
print("total failures", total_failures)
env.close()