import gym
import Box2D
import numpy as np
import cma


def get_action(model, observation):
    actions = np.matmul(model, observation)
    return np.argmax(actions)


def run_env(env, model, render=False):
    env.seed(1339)
    print('start', model, 'end')
    observation = env.reset()
    # print(observation)

    reward_sum = 0
    done = False

    while not done:
        if render:
            env.render()

        action = get_action(model, observation)
        # print(env.step(action))

        observation, reward, done, info = env.step(action)
        reward_sum += reward
        # print(reward)

    # print(reward_sum)
    env.close()
    return reward_sum


def main():
    env = gym.make('LunarLander-v2')

    model_size = env.observation_space.shape[0] \
                 * env.action_space.n

    es = cma.CMAEvolutionStrategy(model_size * [0.0], 1.0)
    best_reward = -10 ** 18
    while not es.stop():

        models = es.ask()
        # print('models', models)
        rewards = []
        for model in models:

            # print('beofre reshape', model)
            model = np.reshape(model, (env.action_space.n, -1))
            # print('step', env.step())
            # print('reset ', env.reset)
            # print('observation ', env.observation_space)

            reward = run_env(env, model)
            rewards.append(-reward)
            if reward > best_reward:
                best_reward = reward
                run_env(env, model, render=True)

        es.tell(models, rewards)
        es.disp()
    env.close()


if __name__ == '__main__':
    main()
