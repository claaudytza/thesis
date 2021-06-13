import numpy
import gym
from qlearnagent import Agent

total_episodes = 50000
max_steps = 100

env = gym.make('Taxi-v3')
qlearn = Agent(env, 0.99, 0.01, 0.9, 0.1, 0.96)

total_rewards, total_penalties = 0, 0

for episode in range(total_episodes):
  obs = env.reset()
  t = 0
  reward_per_episode, penalties = 0, 0
  if episode % 100 == 99:
    qlearn.epsilon *= qlearn.decay_rate
    qlearn.epsilon = max(qlearn.epsilon, qlearn.min_epsilon)
  while t < max_steps:
    action = qlearn.choose_action(obs)
    obs2, reward, done, info = env.step(action)
    if(reward < 0):
      print(reward)
    reward_per_episode += reward
    qlearn.learn(obs, obs2, reward, action)
    obs = obs2
    t += 1
    if done:
        total_rewards += reward_per_episode
        if reward_per_episode == -10:
            penalties += 1
        total_penalties += penalties

print(f"Results after {total_episodes} episodes:")
print(f"Average penalties per episode: {total_penalties / total_episodes}")
print(f"Average rewards per episode: {total_rewards / total_episodes}")
env.close()