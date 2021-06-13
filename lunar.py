import numpy
import gym
from qlearnagent import Agent

total_episodes = 20000
max_steps = 100

env = gym.make('LunarLander-v2')
# env.seed(0)
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
# qlearn = Agent(env, 0.99, 0.01, 0.9, 0.1, 0.96)

# total_success = 0
# total_failures = 0

# for episode in range(total_episodes):
#   obs = env.reset()
#   env.seed(0)
#   rewardsList = []
#   t = 0
#   if episode % 100 == 99:
#     qlearn.epsilon *= qlearn.decay_rate
#     qlearn.epsilon = max(qlearn.epsilon, qlearn.min_epsilon)
#   while t < max_steps:
#     action = qlearn.choose_action(obs)
#     obs2, reward, done, info = env.step(action)
#     qlearn.learn(obs, obs2, reward, action)
#     obs = obs2

#     qlearn.R[obs2, action] = reward
#     rewardsList.append({obs2 : action})

#     t += 1
#     if done:
#       if reward == 100:
#             total_success = total_success + 1
#       else:
#             qlearn.R[obs2, action] = -1
#             if(len(rewardsList) == 3):
#               qlearn.calculateDiscount(rewardsList)
#             total_failures = total_failures + 1
#       break
#     else:
#       if(len(rewardsList) == 3):
#         qlearn.calculateDiscount(rewardsList)
#         rewardsList.pop(0)

# print("total success", total_success)
# print("total failures", total_failures)

# # print(qlearn.R)
# # print(qlearn.Q)
# # print(qlearn.Discount)
# env.close()