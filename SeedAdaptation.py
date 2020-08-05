import numpy as np
import pandas as pd
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
    env = gym.make('LunarLander-v2')
    index_needed = 100000
    data = pd.read_csv('CSVLunarME/gen_{:06d}.csv'.format(index_needed))

    avg_reward_list = []
    # print(data)

    for index in data.index:
        rewards = []
        params = data['model'][index]
        params = filter(lambda x: len(x) > 0, (params.strip('][').split(' ')))
        params = np.array(list(map(float, params)))
        # print(params)
        model = np.reshape(params, (env.action_space.n, -1))
        # print(model)
        num_pos = 0
        # if data['fitness'][index] <= 200:
        #     continue
        for j in range(10):
            env.seed(1400 + j)

            reward, observation = run_env(env, model)
            if reward > 200:
                num_pos += 1
            rewards.append(reward)
        avg_reward = sum(rewards) / len(rewards)
        # print('rewards ', avg_reward, 'data', data['fitness'][index], 'pos', num_pos)
        avg_reward_list.append(avg_reward)
    # data = data[:len(avg_reward_list)]
    data.insert(3, 'avg fitness', avg_reward_list)
    data = pd.DataFrame(data)
    print('saving csv')
    data.to_csv(r"CSVEnvironment\final.csv", index=False, header=True)



if __name__ == '__main__':
    main()
