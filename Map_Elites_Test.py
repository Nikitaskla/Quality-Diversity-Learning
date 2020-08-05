import numpy as np
import cma
import pandas as pd
import cv2
import os
import random
import gym
import Box2D

# num_params = 2

env = gym.make('LunarLander-v2')
num_params = env.observation_space.shape[0] * env.action_space.n


# resolution, highbound, lowbound
class Archive:
    def __init__(self, feature_ranges, resolutions):
        self.feature_ranges = feature_ranges
        self.resolutions = resolutions
        self.elite_map = {}
        self.indices = []

    def add(self, candidate, fitness, features):
        index = self.get_feature_indices(features)
        # print('add func' , index, features)
        if index not in self.elite_map:
            self.indices.append(index)

        if index not in self.elite_map or fitness > self.elite_map[index][1]:
            self.elite_map[index] = (candidate, fitness)

    def get_feature_index(self, dim, b):
        lowbound, highbound = self.feature_ranges[dim]
        if b < lowbound:
            return 0
        if b >= highbound:
            return self.resolutions[dim] - 1
        return int(((b - lowbound) / (highbound - lowbound)) * self.resolutions[dim])

    def get_feature_indices(self, features):
        return tuple(self.get_feature_index(i, feat) for i, feat in enumerate(features))

    def get_random_elite(self):
        index = random.choice(self.indices)
        return self.elite_map[index]

        # images = [img for img in os.listdir(image_folder) if img.endswith(".png")]


class MapElites:
    def __init__(self, init_pop, feature_ranges, resolutions, sigma, num_evaluations):
        self.init_pop = init_pop
        self.solutions_generated = 0
        self.archive = Archive(feature_ranges, resolutions)
        self.sigma = sigma
        self.num_evaluations = num_evaluations

    def ask(self):
        self.solutions_generated += 1
        if self.solutions_generated <= self.init_pop:
            # print(np.random.normal(size=num_params))
            return np.random.normal(size=num_params)

        elite = self.archive.get_random_elite()
        candidate, fitness = elite
        return candidate + np.random.normal(size=num_params, scale=self.sigma)

    def tell(self, candidate, fitness, features):
        self.archive.add(candidate, fitness, features)

    def stop(self):
        return self.solutions_generated >= self.num_evaluations


def get_action(model, observation):
    actions = np.matmul(model, observation)
    # print('model ', model)
    # print('observation ', observation)
    # print('actions', actions)
    # print('arg', np.argmax(actions))
    return np.argmax(actions)


def run_env(env, model, render=False):
    env.seed(1339)
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

    # print(reward_sum)
    env.close()
    if landed_observation is None:
        landed_observation = 8 * [0.0]
    return reward_sum, landed_observation





def main():
    bc_bounds = [(-1.5, 1.5), (-3, 0)]
    resolutions = (100, 100)
    me = MapElites(100, bc_bounds, resolutions, .5, 100000)
    count = 0

    def get_representative(i, index):
        pos = index/resolutions[i]
        val_range = bc_bounds[i]
        delta = val_range[1] - val_range[0]
        return round(pos * delta + val_range[0], 5)


    best_reward = -10 ** 18
    while not me.stop():
        candidate = me.ask()
        rewards = []
        model = np.reshape(candidate, (env.action_space.n, -1))
        reward, observation = run_env(env, model)
        # if reward > best_reward:
        #     best_reward = reward
        #     run_env(env, model, render=True)
        count += 1
        if count % 1000 == 0:
            e = {'x': [x for x, y in me.archive.indices],
                 'y': [y for x, y in me.archive.indices],
                 'fitness': [me.archive.elite_map[index][1] for index in me.archive.indices],
                 'x position': [get_representative(0, x) for x, y in me.archive.indices],
                 'y velocity': [get_representative(1, y) for x, y in me.archive.indices],
                 'model': [me.archive.elite_map[index][0] for index in me.archive.indices]}
            e = pd.DataFrame(e)
            print(count)
            e.to_csv(r"CSVLunarV2\gen_{:06d}.csv".format(count), index=False, header=True)
        # print( 'x_pos', observation[0], 'y_velocity ', observation[3])
        me.tell(candidate, reward, (observation[0], observation[3]))


if __name__ == '__main__':
    main()
