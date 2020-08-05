import gym
import Box2D
import numpy as np

def get_action(model, observation):
    actions = np.matmul(model, observation)
    # print(actions)
    return np.argmax(actions)

def main():
    env = gym.make('LunarLander-v2')
    env.seed(1339)

    observation = env.reset()
    print('observation = ', observation)


    reward_sum = 0
    done = False
    render = False


    # while not done:
    #     env.render()


if __name__ == '__main__':
    main()