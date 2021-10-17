from model import ActorCritic
import torch
import gym
from PIL import Image
import numpy as np

n_episodes = 500

env_name = 'LunarLander-v2'


def test(n_episodes, name='LunarLander_TWO.pth'):
    env = gym.make(env_name)
    state_array = np.ones((n_episodes, 1000, env.observation_space.shape[0]))
    action_array = np.ones((n_episodes, 1000, 1))
    policy = ActorCritic()
    policy.load_state_dict(torch.load('./preTrained/{}'.format(name)))

    render = True
    save_gif = False

    for i_episode in range(0, n_episodes):
        state = env.reset()
        state_array[i_episode, 0] = state
        running_reward = 0
        for t in range(10000):
            action = policy(state)
            state, reward, done, _ = env.step(action)
            state_array[i_episode, 1 + t] = state
            action_array[i_episode, 1 + t] = action
            running_reward += reward
            if render:
                env.render()
                if save_gif:
                    img = env.render(mode='rgb_array')
                    img = Image.fromarray(img)
                    img.save('./gif/{}.jpg'.format(t))
            if done:
                break
        print('Episode {}\tReward: {}'.format(i_episode + 1, running_reward))
    env.close()
    print(state_array)
    print(action_array)
    np.save('data/%s_state_array.npy' % env_name, state_array)
    np.save('data/%s_action_array.npy' % env_name, action_array)


if __name__ == '__main__':
    test(n_episodes=n_episodes)
