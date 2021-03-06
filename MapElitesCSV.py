import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cma
import pandas as pd
import cv2
import os
import random
import csv

num_params = 2


def eval_sphere(x):
    return np.sum(np.square(x - 2.048))


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
        print(index)
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


def main():
    x = np.array([2.048] * num_params)
    # print(x)
    # print(eval_sphere(x))
    count = 0

    me = MapElites(100, [(-5.12, 5.12), (-5.12, 5.12)], (100, 100), .5, 1000)
    while not me.stop():
        candidate = me.ask()
        # print(candidate, 'count=', count)

        fitness = -eval_sphere(candidate)
        # print(fitness)
        count += 1
        if (count % 1000 == 0):
            e = {}
            e['x'] = [x for x, y in me.archive.indices]
            e['y'] = [y for x, y in me.archive.indices]
            e['fitness'] = [me.archive.elite_map[index][1] for index in me.archive.indices]

            e = pd.DataFrame(e)
            # print(e)
            e.to_csv(r"CSV\gen_{:06d}.csv".format(count), index=False, header=True)

        features = candidate
        me.tell(candidate, fitness, features)


if __name__ == '__main__':
    main()
