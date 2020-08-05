import gym
import Box2D
import numpy
from gym import envs

def main():
    env = gym.make('LunarLander-v2')

    env.seed(1333)
    observation = env.reset()
    #print(observation)
    reward_sum = 0
    done = False
    while not done:

        env.render()
        if observation[4] < -.7:
            action = 1
        elif observation[4] > 0.7:
            action = 3
        elif observation[2] > -.3:
            action = 3
        elif observation[0] < 0.0:
            action = 3
        elif observation[0] > 0.0:
            action = 1
        else:
            action = 0

        observation, reward, done, info = env.step(action)
        reward_sum += reward
        print(reward)
    print(reward_sum)
    env.close()

if __name__ == '__main__':
    main()