import os
import math
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import eig
import gym
import Box2D

env = gym.make('LunarLander-v2')
num_params = env.observation_space.shape[0] * env.action_space.n


class Individual:
    pass


class DecompMatrix:
    def __init__(self, dimension):
        self.C = np.eye(dimension, dtype=np.float_)
        self.eigenbasis = np.eye(dimension, dtype=np.float_)
        self.eigenvalues = np.ones((dimension,), dtype=np.float_)
        self.condition_number = 1
        self.invsqrt = np.eye(dimension, dtype=np.float_)

    def update_eigensystem(self):
        for i in range(len(self.C)):
            for j in range(i):
                self.C[i, j] = self.C[j, i]

        self.eigenvalues, self.eigenbasis = eig(self.C)
        self.eigenvalues = np.real(self.eigenvalues)
        self.eigenbasis = np.real(self.eigenbasis)
        self.condition_number = max(self.eigenvalues) / min(self.eigenvalues)

        for i in range(len(self.C)):
            for j in range(i + 1):
                self.invsqrt[i, j] = self.invsqrt[j, i] = sum(
                    self.eigenbasis[i, k] * self.eigenbasis[j, k]
                    / self.eigenvalues[k] ** 0.5 for k in range(len(self.C))
                )


class FeatureMap:

    def __init__(self, max_individuals, feature_ranges, resolutions):
        self.max_individuals = max_individuals
        self.feature_ranges = feature_ranges
        self.resolutions = resolutions

        self.elite_map = {}
        self.elite_indices = []

        self.num_individuals_added = 0

    def get_feature_index(self, feature_id, feature):
        feature_range = self.feature_ranges[feature_id]
        if feature - 1e-9 <= feature_range[0]:
            return 0
        if feature_range[1] <= feature + 1e-9:
            return self.resolutions[feature_id] - 1

        gap = feature_range[1] - feature_range[0]
        pos = feature - feature_range[0]
        index = int((self.resolutions[feature_id] * pos + 1e-9) / gap)
        return index

    def get_index(self, cur):
        return tuple(self.get_feature_index(i, f) for i, f in enumerate(cur.features))

    def add_to_map(self, to_add):
        index = self.get_index(to_add)

        replaced_elite = False
        if index not in self.elite_map:
            self.elite_indices.append(index)
            self.elite_map[index] = to_add
            replaced_elite = True
            to_add.delta = (1, to_add.fitness)
        elif self.elite_map[index].fitness < to_add.fitness:
            to_add.delta = (0, to_add.fitness - self.elite_map[index].fitness)
            self.elite_map[index] = to_add
            replaced_elite = True

        return replaced_elite

    def add(self, to_add):
        self.num_individuals_added += 1
        replaced_elite = self.add_to_map(to_add)
        return replaced_elite

    def get_random_elite(self):
        pos = random.randint(0, len(self.elite_indices) - 1)
        index = self.elite_indices[pos]
        return self.elite_map[index]


class ImprovementEmitter:

    def __init__(self, mutation_power, feature_map):
        self.population_size = int(4.0 + math.floor(3.0 * math.log(num_params))) * 3
        print('pop size', self.population_size)
        self.sigma = mutation_power
        self.individuals_disbatched = 0
        self.individuals_evaluated = 0

        self.parents = []
        self.population = []
        self.feature_map = feature_map

        self.reset()

    def reset(self):
        self.mutation_power = self.sigma
        if len(self.feature_map.elite_map) == 0:
            self.mean = np.asarray([0.0] * num_params)
        else:
            self.mean = self.feature_map.get_random_elite().param_vector

        print('RESET --------------')
        print('new mean:', self.mean)

        # Setup evolution path variables
        self.pc = np.zeros((num_params,), dtype=np.float_)
        self.ps = np.zeros((num_params,), dtype=np.float_)

        # Setup the covariance matrix
        self.C = DecompMatrix(num_params)

        # Reset the individuals evaluated
        self.individuals_evaluated = 0

    def check_stop(self, parents):
        if self.C.condition_number > 1e14:
            return True

        area = self.mutation_power * math.sqrt(max(self.C.eigenvalues))
        if area < 1e-11:
            return True
        if abs(parents[0].fitness - parents[-1].fitness) < 1e-12:
            return True

        return False

    def generate_individual(self):
        unscaled_params = np.random.normal(0.0, self.mutation_power, num_params) * np.sqrt(self.C.eigenvalues)
        unscaled_params = np.matmul(self.C.eigenbasis, unscaled_params)
        unscaled_params = self.mean + unscaled_params
        ind = Individual()
        ind.param_vector = unscaled_params

        self.individuals_disbatched += 1

        return ind

    def return_evaluated_individual(self, ind):
        self.population.append(ind)
        self.individuals_evaluated += 1
        if self.feature_map.add(ind):
            self.parents.append(ind)
        if len(self.population) < self.population_size:
            return

        # Only filter by this generation
        num_parents = len(self.parents)
        needs_restart = num_parents == 0

        # print('parents:', num_parents)

        # Only update if there are parents
        if num_parents > 0:
            parents = sorted(self.parents, key=lambda x: x.delta)[::-1]

            # Create fresh weights for the number of elites found
            weights = [math.log(num_parents + 0.5) \
                       - math.log(i + 1) for i in range(num_parents)]
            total_weights = sum(weights)
            weights = np.array([w / total_weights for w in weights])

            # Dynamically update these parameters
            mueff = sum(weights) ** 2 / sum(weights ** 2)
            cc = (4 + mueff / num_params) / (num_params + 4 + 2 * mueff / num_params)
            cs = (mueff + 2) / (num_params + mueff + 5)
            c1 = 2 / ((num_params + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((num_params + 2) ** 2 + mueff))
            damps = 1 + 2 * max(0, math.sqrt((mueff - 1) / (num_params + 1)) - 1) + cs
            chiN = num_params ** 0.5 * (1 - 1 / (4 * num_params) + 1. / (21 * num_params ** 2))

            # Recombination of the new mean
            old_mean = self.mean
            self.mean = sum(ind.param_vector * w for ind, w in zip(parents, weights))

            # Update the evolution path
            y = self.mean - old_mean
            z = np.matmul(self.C.invsqrt, y)
            self.ps = (1 - cs) * self.ps + \
                      (math.sqrt(cs * (2 - cs) * mueff) / self.mutation_power) * z
            left = sum(x ** 2 for x in self.ps) / num_params \
                   / (1 - (1 - cs) ** (2 * self.individuals_evaluated / self.population_size))
            right = 2 + 4. / (num_params + 1)
            hsig = 1 if left < right else 0

            self.pc = (1 - cc) * self.pc + \
                      hsig * math.sqrt(cc * (2 - cc) * mueff) * y

            # Adapt the covariance matrix
            c1a = c1 * (1 - (1 - hsig ** 2) * cc * (2 - cc))
            self.C.C *= (1 - c1a - cmu)
            self.C.C += c1 * np.outer(self.pc, self.pc)
            for k, w in enumerate(weights):
                dv = parents[k].param_vector - old_mean
                self.C.C += w * cmu * np.outer(dv, dv) / (self.mutation_power ** 2)

            # Update the covariance matrix decomposition and inverse
            if self.check_stop(parents):
                needs_restart = True
            else:
                self.C.update_eigensystem()

            # Update sigma
            cn, sum_square_ps = cs / damps, sum(x ** 2 for x in self.ps)
            self.mutation_power *= math.exp(min(1, cn * (sum_square_ps / num_params - 1) / 2))

        if needs_restart:
            self.reset()

        # Reset the population
        self.population.clear()
        self.parents.clear()


class RandomDirectionEmitter:

    def __init__(self, mutation_power, feature_map):
        self.population_size = int(4.0 + math.floor(3.0 * math.log(num_params))) * 3
        print('pop size', self.population_size)
        self.sigma = mutation_power
        self.individuals_disbatched = 0
        self.individuals_evaluated = 0

        self.parents = []
        self.population = []
        self.feature_map = feature_map
        self.num_features = len(self.feature_map.feature_ranges)

        self.reset()

    def reset(self):
        self.mutation_power = self.sigma
        if len(self.feature_map.elite_map) == 0:
            self.mean = np.asarray([0.0] * num_params)
        else:
            self.mean = self.feature_map.get_random_elite().param_vector
        self.direction = np.asarray([np.random.normal(0.0, 1.0) for _ in range(self.num_features)])

        print('RESET --------------')
        print('new mean:', self.mean, 'direction:', self.direction)

        # Setup evolution path variables
        self.pc = np.zeros((num_params,), dtype=np.float_)
        self.ps = np.zeros((num_params,), dtype=np.float_)

        # Setup the covariance matrix
        self.C = DecompMatrix(num_params)

        # Reset the individuals evaluated
        self.individuals_evaluated = 0

    def check_stop(self, parents):
        if self.C.condition_number > 1e14:
            return True

        area = self.mutation_power * math.sqrt(max(self.C.eigenvalues))
        if area < 1e-11:
            return True
        if abs(parents[0].fitness - parents[-1].fitness) < 1e-12:
            return True

        return False

    def generate_individual(self):
        unscaled_params = np.random.normal(0.0, self.mutation_power, num_params) * np.sqrt(self.C.eigenvalues)
        unscaled_params = np.matmul(self.C.eigenbasis, unscaled_params)
        unscaled_params = self.mean + unscaled_params
        ind = Individual()
        ind.param_vector = unscaled_params

        self.individuals_disbatched += 1

        return ind

    def return_evaluated_individual(self, ind):
        self.population.append(ind)
        self.individuals_evaluated += 1
        if self.feature_map.add(ind):
            self.parents.append(ind)
        if len(self.population) < self.population_size:
            return

        # Update the number of individuals evaluated
        self.individuals_evaluated += 1

        # Only filter by this generation
        num_parents = len(self.parents)
        needs_restart = num_parents == 0

        # Calculate the behavior mean
        feature_mean = sum([np.array(ind.features) for ind in self.population]) / self.population_size
        # print('emitter', ind.emitter_id)
        # print('feature mean', feature_mean)
        # print('parents:', num_parents)

        # Only update if there are parents
        if num_parents > 0:
            for ind in self.parents:
                dv = np.asarray(ind.features) - feature_mean
                ind.projection = np.dot(self.direction, dv)
            parents = sorted(self.parents, key=lambda x: -x.projection)

            # Create fresh weights for the number of elites found
            weights = [math.log(num_parents + 0.5) \
                       - math.log(i + 1) for i in range(num_parents)]
            total_weights = sum(weights)
            weights = np.array([w / total_weights for w in weights])

            # Dynamically update these parameters
            mueff = sum(weights) ** 2 / sum(weights ** 2)
            cc = (4 + mueff / num_params) / (num_params + 4 + 2 * mueff / num_params)
            cs = (mueff + 2) / (num_params + mueff + 5)
            c1 = 2 / ((num_params + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((num_params + 2) ** 2 + mueff))
            damps = 1 + 2 * max(0, math.sqrt((mueff - 1) / (num_params + 1)) - 1) + cs
            chiN = num_params ** 0.5 * (1 - 1 / (4 * num_params) + 1. / (21 * num_params ** 2))

            # Recombination of the new mean
            old_mean = self.mean
            self.mean = sum(ind.param_vector * w for ind, w in zip(parents, weights))

            # Update the evolution path
            y = self.mean - old_mean
            z = np.matmul(self.C.invsqrt, y)
            self.ps = (1 - cs) * self.ps + \
                      (math.sqrt(cs * (2 - cs) * mueff) / self.mutation_power) * z
            left = sum(x ** 2 for x in self.ps) / num_params \
                   / (1 - (1 - cs) ** (2 * self.individuals_evaluated / self.population_size))
            right = 2 + 4. / (num_params + 1)
            hsig = 1 if left < right else 0

            self.pc = (1 - cc) * self.pc + \
                      hsig * math.sqrt(cc * (2 - cc) * mueff) * y

            # Adapt the covariance matrix
            c1a = c1 * (1 - (1 - hsig ** 2) * cc * (2 - cc))
            self.C.C *= (1 - c1a - cmu)
            self.C.C += c1 * np.outer(self.pc, self.pc)
            for k, w in enumerate(weights):
                dv = parents[k].param_vector - old_mean
                self.C.C += w * cmu * np.outer(dv, dv) / (self.mutation_power ** 2)

            # Update the covariance matrix decomposition and inverse
            if self.check_stop(parents):
                needs_restart = True
            else:
                self.C.update_eigensystem()

            # Update sigma
            cn, sum_square_ps = cs / damps, sum(x ** 2 for x in self.ps)
            self.mutation_power *= math.exp(min(1, cn * (sum_square_ps / num_params - 1) / 2))

        if needs_restart:
            self.reset()

        # Reset the population
        self.population.clear()
        self.parents.clear()


class OptimizingEmitter:

    def __init__(self, mutation_power, feature_map):
        self.population_size = int(4.0 + math.floor(3.0 * math.log(num_params)))
        print('pop size', self.population_size)
        self.sigma = mutation_power
        self.individuals_disbatched = 0
        self.individuals_evaluated = 0

        self.population = []
        self.feature_map = feature_map

        self.reset()

    def reset(self):
        self.mutation_power = self.sigma
        if len(self.feature_map.elite_map) == 0:
            self.mean = np.asarray([0.0] * num_params)
        else:
            self.mean = self.feature_map.get_random_elite().param_vector

        print('RESET --------------')
        print('new mean:', self.mean)

        # Setup evolution path variables
        self.pc = np.zeros((num_params,), dtype=np.float_)
        self.ps = np.zeros((num_params,), dtype=np.float_)

        # Setup the covariance matrix
        self.C = DecompMatrix(num_params)

        # Reset the individuals evaluated
        self.individuals_evaluated = 0

    def check_stop(self, parents):
        if self.C.condition_number > 1e14:
            return True

        area = self.mutation_power * math.sqrt(max(self.C.eigenvalues))
        if area < 1e-11:
            return True
        if abs(parents[0].fitness - parents[-1].fitness) < 1e-12:
            return True

        return False

    def generate_individual(self):
        unscaled_params = np.random.normal(0.0, self.mutation_power, num_params) * np.sqrt(self.C.eigenvalues)
        unscaled_params = np.matmul(self.C.eigenbasis, unscaled_params)
        unscaled_params = self.mean + unscaled_params
        ind = Individual()
        ind.param_vector = unscaled_params

        self.individuals_disbatched += 1

        return ind

    def return_evaluated_individual(self, ind):
        self.population.append(ind)
        self.feature_map.add(ind)
        if len(self.population) < self.population_size:
            return

        # Update the number of individuals evaluated
        self.individuals_evaluated += 1

        # Only update if there are parents
        num_parents = self.population_size // 2
        parents = sorted(self.population, key=lambda x: x.fitness)[::-1]
        parents = parents[:num_parents]
        print('----', parents[0].fitness)

        # Create fresh weights for the number of elites found
        weights = [math.log(num_parents + 0.5) \
                   - math.log(i + 1) for i in range(num_parents)]
        total_weights = sum(weights)
        weights = np.array([w / total_weights for w in weights])

        # Dynamically update these parameters
        mueff = sum(weights) ** 2 / sum(weights ** 2)
        cc = (4 + mueff / num_params) / (num_params + 4 + 2 * mueff / num_params)
        cs = (mueff + 2) / (num_params + mueff + 5)
        c1 = 2 / ((num_params + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((num_params + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, math.sqrt((mueff - 1) / (num_params + 1)) - 1) + cs
        chiN = num_params ** 0.5 * (1 - 1 / (4 * num_params) + 1. / (21 * num_params ** 2))

        # Recombination of the new mean
        old_mean = self.mean
        self.mean = sum(ind.param_vector * w for ind, w in zip(parents, weights))

        # Update the evolution path
        y = self.mean - old_mean
        z = np.matmul(self.C.invsqrt, y)
        self.ps = (1 - cs) * self.ps + \
                  (math.sqrt(cs * (2 - cs) * mueff) / self.mutation_power) * z
        left = sum(x ** 2 for x in self.ps) / num_params \
               / (1 - (1 - cs) ** (2 * self.individuals_evaluated / self.population_size))
        right = 2 + 4. / (num_params + 1)
        hsig = 1 if left < right else 0

        self.pc = (1 - cc) * self.pc + \
                  hsig * math.sqrt(cc * (2 - cc) * mueff) * y

        # Adapt the covariance matrix
        c1a = c1 * (1 - (1 - hsig ** 2) * cc * (2 - cc))
        self.C.C *= (1 - c1a - cmu)
        self.C.C += c1 * np.outer(self.pc, self.pc)
        for k, w in enumerate(weights):
            dv = parents[k].param_vector - old_mean
            self.C.C += w * cmu * np.outer(dv, dv) / (self.mutation_power ** 2)

        # Update the covariance matrix decomposition and inverse
        needs_restart = self.check_stop(parents)
        if not needs_restart:
            self.C.update_eigensystem()

        # Update sigma
        cn, sum_square_ps = cs / damps, sum(x ** 2 for x in self.ps)
        self.mutation_power *= math.exp(min(1, cn * (sum_square_ps / num_params - 1) / 2))

        if needs_restart:
            self.reset()

        # Reset the population
        self.population.clear()


class CMA_ME_Algorithm:

    def __init__(self, mutation_power, initial_population, num_to_evaluate, feature_map):
        self.emitters = None
        self.initial_population = initial_population
        self.num_to_evaluate = num_to_evaluate
        self.individuals_disbatched = 0
        self.individuals_evaluated = 0
        self.feature_map = feature_map
        self.mutation_power = mutation_power

    def is_running(self):
        return self.individuals_evaluated < self.num_to_evaluate

    def generate_individual(self):
        ind = None
        if self.individuals_disbatched < self.initial_population:
            ind = Individual()
            if self.individuals_evaluated < self.initial_population:
                unscaled_params = np.random.normal(0.0, 1.0, num_params)
                ind.param_vector = unscaled_params
            ind.emitter_id = -1
        else:
            if self.emitters is None:
                self.emitters = []
                # self.emitters += [RandomDirectionEmitter(self.mutation_power, self.feature_map) for i in range(1)]
                self.emitters += [ImprovementEmitter(self.mutation_power, self.feature_map) for i in range(15)]
                # self.emitters += [OptimizingEmitter(self.mutation_power, self.feature_map) for i in range(1)]

            pos = 0
            emitter = self.emitters[0]
            for i in range(1, len(self.emitters)):
                if self.emitters[i].individuals_disbatched < emitter.individuals_disbatched:
                    emitter = self.emitters[i]
                    pos = i
            ind = emitter.generate_individual()
            ind.emitter_id = pos

        self.individuals_disbatched += 1
        return ind

    def return_evaluated_individual(self, ind):
        ind.ID = self.individuals_evaluated
        self.individuals_evaluated += 1

        if ind.emitter_id == -1:
            self.feature_map.add(ind)
            print('ADD')
        else:
            self.emitters[ind.emitter_id].return_evaluated_individual(ind)


def run_cma_me(num_to_evaluate, initial_population, mutation_power=0.5):
    resolution = (100, 100)
    feature_ranges = [(-1.5, 1.5), (-3, 0)]
    feature_map = FeatureMap(num_to_evaluate, feature_ranges, resolution)

    cma_me = CMA_ME_Algorithm(mutation_power,
                              initial_population,
                              num_to_evaluate,
                              feature_map)

    best = -10 ** 18
    count = 0

    def get_representative(i, index):
        pos = index / resolution[i]
        val_range = feature_ranges[i]
        delta = val_range[1] - val_range[0]
        return round(pos * delta + val_range[0], 5)

    while cma_me.is_running():
        ind = cma_me.generate_individual()
        model = np.reshape(ind.param_vector, (env.action_space.n, -1))
        ind.fitness, observation = run_env(env, model)  # CHANGE_LUNAR
        ind.features = (observation[0], observation[3])
        if ind.fitness > best:
            best = ind.fitness
            print(ind.fitness)
        count += 1
        if count % 100 == 0:
            print('individuals evaluated', count)

        if count % 1000 == 0:
            e = {'x': [x for x, y in feature_map.elite_indices],
                 'y': [y for x, y in feature_map.elite_indices],
                 'fitness': [feature_map.elite_map[index].fitness for index in feature_map.elite_indices],
                 'x position': [get_representative(0, x) for x, y in feature_map.elite_indices],
                 'y velocity': [get_representative(1, y) for x, y in feature_map.elite_indices],
                 'model': [feature_map.elite_map[index].param_vector for index in feature_map.elite_indices]}
            e = pd.DataFrame(e)
            print('saving csv')
            e.to_csv(r"CSVLunarME\gen_{:06d}.csv".format(count), index=False, header=True)
        cma_me.return_evaluated_individual(ind)
    # print(best)


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


# run_cma_es(50000, num_parents=250, population_size=500, mutation_power=0.8)
run_cma_me(100000, 0, mutation_power=0.05)
# run_map_elites(50000, 100, mutation_power=0.05)
