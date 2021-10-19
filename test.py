import numpy as np
import torch

seeds = [684, 559, 629, 192, 835]  # random seeds for testing
seed_reward = []
gamma = 1.0  # discount factor

def test(env, model, n_iterations=5 ,n_ep = 20, max_steps = 500):
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
                action = model(torch.tensor(state, dtype=torch.float))
                action = torch.argmax(action).item()
                # action = np.clip(action.detach().numpy(), -1, 1)
                # state_array[ep, t] = state
                # action_array[ep, t] = action
                next_state, r, done, _ = env.step(action)
                env.render()
                # time.sleep(0.01)
                rewards.append(r)
                state = next_state
                if done:
                    break
            R = sum([rewards[i] * gamma ** i for i in range(len(rewards))])
            G.append(R)
        seed_reward.append(G)
        print("Itr = {} overall reward  = {:.6f} ".format(itr, np.mean(seed_reward[-1])))
        print("Interacting with environment finished")
    return seed_reward