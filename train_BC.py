import numpy as np
import torch
from torch import nn as nn

loss_list = []
test_losses = []
test_acc = []
learning_rate = 0.003
dic = {}


def train_BC(args, policy, batch_size=256, n_epoch=5):
    '''
    train Behavioral Cloning model, given pair of states return action (s0,s1 ---> a0 if n=2)
    Input:
    training_set:
    policy: Behavioral Cloning model want to train
    n: window size (how many states needed to predict the next action)
    batch_size: batch size
    n_epoch: number of epoches
    return:
    policy: trained Behavioral Cloning model
    '''
    criterion = nn.MSELoss()
    # criterion = nn.CrossEntropyLoss()
    training_set = torch.tensor(args['training_data'], dtype=torch.float)
    testing_set = torch.tensor(args['testing_data'], dtype=torch.float)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    for itr in range(n_epoch):
        total_loss = 0
        total_item = 0
        total_right = 0
        b = 0
        for batch in range(0, training_set.shape[0], batch_size):
            np.random.shuffle(args['training_data'])
            training_set = torch.tensor(args['training_data'], dtype=torch.float)
            data = training_set[batch: batch + batch_size, :args['state_space_size']]
            y = training_set[batch: batch + batch_size, args['state_space_size']:]
            y = y.squeeze(1).to(torch.int64)
            y_one_hot = torch.nn.functional.one_hot(y, args['action_space_size'])
            y_pred = policy(data)
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
        x = testing_set[:, : args['state_space_size']]
        y = testing_set[:, args['state_space_size']:]
        y = y.squeeze(1).to(torch.int64)
        y_one_hot = torch.nn.functional.one_hot(y, args['action_space_size'])
        y_pred = policy(x)
        test_loss = criterion(y_pred, y_one_hot.to(torch.float))
        # test_loss = criterion(y_pred, y)
        test_losses.append(test_loss.item())
        test_acc.append(torch.eq(y, torch.argmax(y_pred, dim=1)).to(torch.float).mean().item())
    dic['train_loss'] = loss_list
    dic['test_loss'] = test_losses
    dic['test_acc'] = test_acc

    return dic
