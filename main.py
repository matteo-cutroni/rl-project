from models import Manager, Controller, Imagination, Memory
import gymnasium as gym
import torch
import torch.nn.functional as F
import numpy as np

def train(episodes, n_max_real_steps, n_max_imagined_steps, gamma):

    env = gym.make('Blackjack-v1', natural=False, sab=False)

    state_size = len(env.observation_space)
    action_size = env.action_space.n
    history_size = state_size*2 + 5  #x_real, x_imagined, a, u, n_real, n_imagined, reward

    manager = Manager(state_size, history_size)
    controller = Controller(state_size, history_size, action_size)
    imagination = Imagination(state_size)
    memory = Memory(history_size, history_size)

    for ep in range(episodes):
        s, _ = env.reset()
        s = torch.tensor(s)

        history = torch.zeros(history_size)
        n_real, n_imagined = 0, 0
        x_real, x_imagined = s, s

        real_states, imagined_states = [], []
        real_rewards, imagined_rewards = [], []

        done, truncated = False, False

        while n_real < n_max_real_steps:
            u = torch.argmax(manager(x_real, history)).unsqueeze(0)
            print(u)

            if u == 0 or n_imagined > n_max_imagined_steps:
                print('taking action')

                a = torch.argmax(controller(x_real, history)).unsqueeze(0)

                x_real, r, done, truncated, _ = env.step(a.item())
                x_real = torch.tensor(x_real)
                r = torch.tensor([r])

                real_states.append(x_real)
                real_rewards.append(r)

                s_im, r_im = imagination(x_real, a) #for imagination training
                imagined_states.append(s_im)
                imagined_rewards.append(r_im)

                n_real += 1
                n_imagined = 0
                x_imagined = x_real

                print(done, truncated)

            elif u == 1:
                print('imagining', n_imagined)
                a = torch.argmax(controller(x_real, history)).unsqueeze(0)
                x_imagined, r = imagination(x_real, a)
                n_imagined += 1

            elif u == 2:
                print('imagining', n_imagined)
                a = torch.argmax(controller(x_real, history)).unsqueeze(0)
                x_imagined, r = imagination(x_imagined, a)
                n_imagined += 1

            d = torch.cat([u, a, r, x_real, x_imagined, torch.tensor([n_real]), torch.tensor([n_imagined])]).unsqueeze(0)
            history = memory(d)
        
            if done or truncated:
                print(r)
                break

        #TODO arrays to tensor    
        imagination_loss = F.mse_loss(imagined_states, real_states) + F.mse_loss(imagined_rewards, real_rewards)

        manager_loss = 0
        rewards = real_rewards + imagined_rewards
        print(rewards)
        discounts = np.power(gamma, np.arange(len(rewards)))





if __name__ == '__main__':
    episodes = 10
    n_max_real_steps = 5
    n_max_imagined_steps = 5
    gamma = 0.9

    train(episodes, n_max_real_steps, n_max_imagined_steps, gamma)