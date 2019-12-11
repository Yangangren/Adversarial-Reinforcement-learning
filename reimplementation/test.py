import gym
import numpy as np

env = gym.make('CentralDecisionTest-v0')
for episode in range(5):
    observation = env.reset()
    for t in range(1000):
        env.render()
        # print(observation)
        action = env.action_space.sample()
        adv_action = env.adv_action_space.sample()
        action = np.append(action, adv_action)
        observation, reward, done, info = env.step(action)
        if done:
            print('Episode finished after {} time steps'.format(t + 1))
            break
env.close()