import random
import time

import numpy as np
import gym
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from IPython import display
from matplotlib import style

from model.net import Net
from test import test
from utils import to_input

plt.style.use("ggplot")

# init environment
number_expert_trajectories = 200
# env_name = 'MountainCar-v0'
env_name = 'LunarLander-v2'
env = gym.make(env_name)
action_space_size = env.action_space.shape[0] if len(env.action_space.shape) > 0 else 4
state_space_size = env.observation_space.shape[0]
# locate = None
locate = './model/%s_%s.pkl' % (env_name, number_expert_trajectories)

# Load Expert data (states and actions for BC, States only for BCO)
expert_states = torch.tensor(np.load("expert/data/%s_state_array.npy" % env_name), dtype=torch.float)
expert_actions = torch.tensor(np.load("expert/data/%s_action_array.npy" % env_name), dtype=torch.float)
print("expert_states", expert_states.shape)
print("expert_actions", expert_actions.shape)

# Network arch Behavioral Cloning , loss function and optimizer
bc_walker = Net(state_space_size, action_space_size)

if locate is None:
    # selecting number expert trajectories from expert data
    a = np.random.randint(1 + expert_states.shape[0] - number_expert_trajectories)
    expert_state, expert_action = to_input(expert_states[a: a + number_expert_trajectories],
                                           expert_actions[a: a + number_expert_trajectories],
                                           n=2)
    print("expert_state", expert_state.shape)
    print("expert_action", expert_action.shape)

    new_data = np.concatenate((expert_state[:, : state_space_size], expert_action), axis=1)
    np.random.shuffle(new_data)
    # new_data = torch.tensor(new_data, dtype=torch.float)
    n_samples = int(new_data.shape[0] * 0.8)
    training_data = new_data[:n_samples]
    testing_data = new_data[n_samples:]
    training_set = torch.tensor(training_data, dtype=torch.float)
    testing_set = torch.tensor(testing_data, dtype=torch.float)

    print("training_set", training_set.shape)
    print("testing_set", testing_set.shape)
    p = 87  # select any point to test the model
    print(bc_walker(testing_set[p, :state_space_size]))
    print(testing_set[p, state_space_size:])
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    learning_rate = 0.003

    loss_list = []
    test_loss = []
    test_acc = []
    batch_size = 256
    n_epoch = 50
    optimizer = torch.optim.Adam(bc_walker.parameters(), lr=learning_rate)
    for itr in range(n_epoch):
        total_loss = 0
        total_item = 0
        total_right = 0
        b = 0
        for batch in range(0, training_set.shape[0], batch_size):
            np.random.shuffle(training_data)
            training_set = torch.tensor(training_data, dtype=torch.float)
            data = training_set[batch: batch + batch_size, :state_space_size]
            y = training_set[batch: batch + batch_size, state_space_size:]
            y = y.squeeze(1).to(torch.int64)
            y_one_hot = torch.nn.functional.one_hot(y, action_space_size)
            y_pred = bc_walker(data)
            loss = criterion(y_pred, y_one_hot.to(torch.float))
            # loss = criterion(y_pred, y)
            total_loss += loss.item()
            total_item += batch_size
            total_right += torch.eq(y, torch.argmax(y_pred, dim=1)).to(torch.float).sum().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            b += 1
        print("[EPOCH]: %i, [CE LOSS]: %.6f, [ACC]: %f" % (itr + 1, total_loss / b, total_right / total_item))
        # display.clear_output(wait=True)
        loss_list.append(total_loss / b)
        x = testing_set[:, :state_space_size]
        y = testing_set[:, state_space_size:]

        y = y.squeeze(1).to(torch.int64)
        y_one_hot = torch.nn.functional.one_hot(y, action_space_size)
        y_pred = bc_walker(x)
        loss = criterion(y_pred, y_one_hot.to(torch.float))
        # loss = criterion(y_pred, y)
        test_loss.append(loss.item())
        test_acc.append(torch.eq(y, torch.argmax(y_pred, dim=1)).to(torch.float).mean().item())

    torch.save(bc_walker, './model/%s_%s.pkl' % (env_name, number_expert_trajectories))
    # plot test loss
    # torch.save(bc_walker, "bc_walker_n=2") # uncomment to save the model
    # plt.subplot(1,1)
    plt.plot(test_loss, label="Testing Loss")
    plt.xlabel("iterations")
    plt.ylabel("loss")

    # plt.subplot(1,2)
    plt.plot(test_acc, label="Testing ACC")
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
