import numpy as np
import gym
import torch
from matplotlib import pyplot as plt
from model.net import Net
from test import test
from train_BC import train_BC
from utils import to_input

plt.style.use("ggplot")

# init environment
number_expert_trajectories = 200
env_name = 'LunarLander-v2' # 'MountainCar-v0'
env = gym.make(env_name)
action_space_size = env.action_space.shape[0] if len(env.action_space.shape) > 0 else 4
state_space_size = env.observation_space.shape[0]
locate = None
# locate = './model/%s_%s.pkl' % (env_name, number_expert_trajectories)

expert_states = torch.tensor(np.load("expert/data/%s_state_array.npy" % env_name), dtype=torch.float)
expert_actions = torch.tensor(np.load("expert/data/%s_action_array.npy" % env_name), dtype=torch.float)

bc_walker = Net(state_space_size, action_space_size)

if locate is None:
    # selecting number expert trajectories from expert data
    a = np.random.randint(1 + expert_states.shape[0] - number_expert_trajectories)
    expert_state, expert_action = to_input(expert_states[a: a + number_expert_trajectories],
                                           expert_actions[a: a + number_expert_trajectories],
                                           n=2)
    new_data = np.concatenate((expert_state[:, : state_space_size], expert_action), axis=1)
    print("expert_state", expert_state.shape)
    print("expert_action", expert_action.shape)
    np.random.shuffle(new_data)
    n_samples = int(new_data.shape[0] * 0.8)
    training_data = new_data[:n_samples]
    testing_data = new_data[n_samples:]

    args = {'training_data': training_data, 'testing_data': testing_data, 'state_space_size': state_space_size,
            'action_space_size': action_space_size}

    dic = train_BC(args, bc_walker)
    # torch.save(bc_walker, './model/%s_%s.pkl' % (env_name, number_expert_trajectories))


    plt.plot(dic['test_loss'], label="Testing Loss")
    plt.xlabel("iterations")
    plt.ylabel("loss")

    plt.plot(dic['test_acc'], label="Testing ACC")
    plt.xlabel("iterations")
    plt.ylabel("acc")
    plt.legend()
    plt.show()
else:
    bc_walker = torch.load(locate)
# ## Test inferred actions with real actions

seed_reward = test(env, bc_walker, n_iterations=5, n_ep=20)

env.close()
# np.save("reward_mean_walker_bc1_expert_states={}".format(new_data.shape[0]), seed_reward_mean) #uncomment to save reward over 5 random seeds

seed_reward_mean_bc = np.array(seed_reward)
mean_bc = np.mean(seed_reward_mean_bc, axis=0)
std_bc = np.std(seed_reward_mean_bc, axis=0)

# plt.plot(x, mean_bc, "-", label="BC")
# plt.fill_between(x, mean_bc + std_bc, mean_bc - std_bc, alpha=0.2)

plt.plot(mean_bc, "-", label="BC")
plt.xlabel("Episodes")
plt.ylabel("Mean Reward")

plt.legend()
plt.show()
