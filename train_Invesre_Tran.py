import numpy as np
import torch
from torch import nn as nn

loss_list = []
test_losses = []
test_acc = []
learning_rate = 0.001

dic = {}


def train_BC(args, state_trainsition_model, batch_size=128, n_epoch=500):
    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(state_trainsition_model.parameters(), lr=learning_rate)
    training_set = torch.tensor(args['training_data'], dtype=torch.float)
    testing_set = torch.tensor(args['testing_data'], dtype=torch.float)
    n = args['stack']

    for itr in range(n_epoch):
        total_loss = 0
        total_item = 0
        total_right = 0
        b = 0
        for batch in range(0, training_set.shape[0], batch_size):
            np.random.shuffle(args['training_data'])
            training_set = torch.tensor(args['training_data'], dtype=torch.float)
            x = training_set[batch: batch + batch_size, :n * args['state_space_size']]
            y = training_set[batch: batch + batch_size, n * args['state_space_size']:]
            y = y.squeeze(1).to(torch.int64)
            y_one_hot = torch.nn.functional.one_hot(y, args['action_space_size'])
            y_pred = state_trainsition_model(x)
            loss = criterion(y_pred, y_one_hot.to(torch.float))
            total_loss += loss.item()
            total_item += batch_size
            total_right += torch.eq(y, torch.argmax(y_pred, dim=1)).to(torch.float).sum().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            b += 1
        print("[EPOCH]: %i, [LOSS]: %.6f" % (itr + 1, total_loss / b))

        loss_list.append(total_loss / b)
        x = testing_set[:, :n * args['state_space_size']]
        y = testing_set[:, n * args['state_space_size']:]
        y = y.squeeze(1).to(torch.int64)
        y_one_hot = torch.nn.functional.one_hot(y, args['action_space_size'])
        y_pred = state_trainsition_model(x)
        test_loss = criterion(y_pred, y_one_hot.to(torch.float))
        # test_loss = criterion(y_pred, y).item()
        test_losses.append(test_loss.item())
        test_acc.append(torch.eq(y, torch.argmax(y_pred, dim=1)).to(torch.float).mean().item())
    dic['train_loss'] = loss_list
    dic['test_loss'] = test_losses
    dic['test_acc'] = test_acc

    return dic
