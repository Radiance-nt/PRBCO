import torch


def to_input(states, actions, n=2, compare=-99):
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
