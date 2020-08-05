import numpy as np
import pandas as pd
import cv2
import os
import gym
import Box2D


def get_action(model, observation):
    actions = np.matmul(model, observation)
    # print('model ', model)
    # print('observation ', observation)
    # print('actions', actions)
    # print('arg', np.argmax(actions))
    return np.argmax(actions)


def run_env(env, model, render=False):
    env.seed(1498)
    # print(model)
    observation = env.reset()
    # print(observation)

    reward_sum = 0
    done = False
    landed_observation = None
    while not done:
        if render:
            env.render()

        action = get_action(model, observation)
        # print(env.step(action))

        observation, reward, done, info = env.step(action)
        # print(observation)
        reward_sum += reward
        # print(observation)
        if landed_observation is None and (observation[6] == 1 or observation[7] == 1):
            landed_observation = observation


def main():
    env = gym.make('LunarLander-v2')
    # env = gym.wrappers.Monitor(env, "./vid", video_callable=lambda episode_id: True, force=True)
    index_needed = 66000

    model_size = env.observation_space.shape[0] * env.action_space.n
    #    params = '[0.78298982  1.04000919  2.57045781  1.21016969  0.3440427   0.23668062 \
    #  0.83362875  0.76388967 -1.81892483  0.48111372 -0.12253041 -0.17428457 \
    #  0.87581401  0.11000786  0.26591139 -2.45817028  2.12107001  0.35629291 \
    #  0.22560889  1.13086624  1.45361522  0.00581429 -1.83497918 -0.9242057 \
    # -0.30153263 -0.46900917  0.93313228  0.36368871  0.06223366 -0.97098315 \
    #  0.57865239 -0.01734878]'
    #
    #

    data = pd.read_csv('CSVEnvironment/final.csv')
    # print(data)
    dataylim = data
    # print('ylim', dataylim)

    elite_index = dataylim['fitness'].argmax()
    # print(elite_index)
    elite = dataylim.iloc[elite_index]
    print(elite)
    params = elite['model']

    params = filter(lambda x: len(x) > 0, (params.strip('][').split(' ')))
    params = list(map(float, params))
    print(params)

    model = np.reshape(params, (env.action_space.n, -1))
    reward, observation = run_env(env, model, render=True)

    # data = pd.read_csv('CSVLunarV2/gen_{:06d}.csv'.format(count))


if __name__ == '__main__':
    main()
