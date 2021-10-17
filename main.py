import random

import numpy as np
import gym
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from IPython import display
from matplotlib import style

from model import net

plt.style.use("ggplot")

# init environment
env_name = 'LunarLander-v2'
env = gym.make(env_name)
action_space_size = env.action_space.shape[0] if len(env.action_space.shape) > 0 else 4
state_space_size = env.observation_space.shape[0]

# Load Expert data (states and actions for BC, States only for BCO)
expert_states = torch.tensor(np.load("expert/data/%s_state_array.npy" % env_name), dtype=torch.float)
expert_actions = torch.tensor(np.load("expert/data/%s_action_array.npy" % env_name), dtype=torch.float)
print("expert_states", expert_states.shape)
print("expert_actions", expert_actions.shape)


# In[5]:


def to_input(states, actions, n=2, compare=1):
    '''
    Data preperpation and filtering
    Inputs:
    states: expert states as tensor
    actions: actions states as tensor
    n: window size (how many states needed to predict the next action)
    compare: for filtering data
    return:
    output_states: filtered states as tensor
    output_actions: filtered actions as tensor
    '''
    count = 0
    index = []
    ep, t, state_size = states.shape
    _, _, action_size = actions.shape

    output_states = torch.zeros((ep * (t - n + 1), state_size * n), dtype=torch.float)
    output_actions = torch.zeros((ep * (t - n + 1), action_size), dtype=torch.float)

    for i in range(ep):
        for j in range(t - n + 1):
            if (states[i, j] == -compare * torch.ones(state_size)).all() or (
                    states[i, j + 1] == -compare * torch.ones(state_size)).all():
                index.append([i, j])
            else:
                output_states[count] = states[i, j:j + n].view(-1)
                output_actions[count] = actions[i, j]
                count += 1
    output_states = output_states[:count]
    output_actions = output_actions[:count]

    return output_states, output_actions


# In[6]:


# selecting number expert trajectories from expert data
number_expert_trajectories = 400
a = np.random.randint(expert_states.shape[0] - number_expert_trajectories)
expert_state, expert_action = to_input(expert_states[a: a + number_expert_trajectories],
                                       expert_actions[a: a + number_expert_trajectories],
                                       n=2, compare=-1)
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


# Network arch Behavioral Cloning , loss function and optimizer
bc_walker = nn.Sequential(
    nn.Linear(state_space_size, 40),
    nn.ReLU(),

    # nn.Linear(40, 80),
    # nn.ReLU(),
    #
    # nn.Linear(80, 120),
    # nn.ReLU(),
    #
    # nn.Linear(120, 100),
    # nn.ReLU(),
    #
    # nn.Linear(100, 40),
    # nn.ReLU(),

    nn.Linear(40, 20),
    nn.ReLU(),

    nn.Linear(20, action_space_size),
    nn.Softmax()
)
locate = None
# locate = "model/100.pkl"
if locate is None:
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    learning_rate = 0.005

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
            total_item += torch.eq(y, torch.argmax(y_pred, dim=1)).to(torch.float).shape[0]
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

    torch.save(bc_walker, './model/%s.pkl' % number_expert_trajectories)
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

p = 87  # select any point to test the model
print(bc_walker(testing_set[p, :state_space_size]))
print(testing_set[p, state_space_size:])
# criterion(bc_walker(testing_set[p, :state_space_size]), testing_set[p, state_space_size:]).item()

################################## parameters ##################################
n = 2  # window size
n_iterations = 5  # max number of interacting with environment
n_ep = 20  # number of epoches
max_steps = 500  # max timesteps per epoch
gamma = 1.0  # discount factor
seeds = [684, 559, 629, 192, 835]  # random seeds for testing
################################## parameters ##################################

seed_reward = []
for itr in range(n_iterations):
    ################################## interact with env ##################################
    G = []
    G_mean = []
    env.seed(int(seeds[itr]))
    torch.manual_seed(int(seeds[itr]))
    torch.cuda.manual_seed_all(int(seeds[itr]))

    for ep in range(n_ep):
        state = env.reset()
        rewards = []
        R = 0
        for t in range(max_steps):
            action = bc_walker(torch.tensor(state, dtype=torch.float))
            action = torch.argmax(action).item()
            # action = np.clip(action.detach().numpy(), -1, 1)
            # action = env.action_space.sample()
            next_state, r, done, _ = env.step(action)
            env.render()
            rewards.append(r)
            state = next_state
            if done:
                break
        R = sum([rewards[i] * gamma ** i for i in range(len(rewards))])
        G.append(R)
    seed_reward.append(G)
    print("Itr = {} overall reward  = {:.6f} ".format(itr, np.mean(seed_reward[-1])))
    print("Interacting with environment finished")
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
