import random
import gym
import Box2D
import numpy
from gym import envs

env = gym.make('LunarLander-v2')
#print("Observation space: ", env.observation_space)
#print("Action space: ", env.action_space)

"""
class MapElites():
    {
        numpy.zeros((25, 25))
        i=0
        for x in range(5)
            if i < G:
            

    }
"""

class Agent():
    def __init__(self, env):
        self.action_size = env.action_space.n
        print("Action size: ", self.action_size)

    def get_action(self):


        act = random.choice(range(self.action_size))
        print(act)
        return act


lander = Agent(env)
state = env.reset()

for _ in range(200):
        action = env.action_space.sample()
        action = lander.get_action()
        state, reward, done, idk = env.step(action)
        print("state = ", state, "reward = ", reward, "IfDone = ", done, "?? = ", idk)
        env.render()

env.close()

