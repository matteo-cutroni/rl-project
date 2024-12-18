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

    imagination_optim = torch.optim.Adam(imagination.parameters(), lr=1e-3)
    manager_optim = torch.optim.Adam(manager.parameters(), lr=1e-3)
    con_mem_optim = torch.optim.Adam(list(controller.parameters())+list(memory.parameters()), lr=1e-3)

    manager.train()
    controller.train()
    imagination.train()
    memory.train()

    for ep in range(episodes):
        s, _ = env.reset()
        s = torch.tensor(s).float()

        n_real, n_imagined = 0, 0
        x_real, x_imagined = s, s

        history = torch.zeros(history_size)

        real_states, imagined_states = [], []
        real_rewards, imagined_rewards = [], []

        external_log_probs_manager = []
        tot_imagination_steps = 0

        log_probs_controller = []
        tot_rewards = []

        done, truncated = False, False

        while n_real < n_max_real_steps:
            u_probs = manager(x_real, history)
            manager_log_probs = torch.log(u_probs)
            u = torch.argmax(u_probs).unsqueeze(0)

            if u == 0 or n_imagined > n_max_imagined_steps:
                external_log_probs_manager.append(manager_log_probs[0])

                a_probs = controller(x_real, history)
                a = torch.argmax(a_probs).unsqueeze(0)
                log_probs_controller.append(torch.log(a_probs[a]))
                a = torch.tensor(a).float()

                x_real, r, done, truncated, _ = env.step(int(a.item()))
                x_real = torch.tensor(x_real).float()
                r = torch.tensor([r]).float()
                tot_rewards.append(r)

                real_states.append(x_real)
                real_rewards.append(r)

                s_im, r_im = imagination(x_real, a) #for imagination training
                imagined_states.append(s_im)
                imagined_rewards.append(r_im)

                n_real += 1
                n_imagined = 0
                x_imagined = x_real

            elif u == 1:
                a_probs = controller(x_real, history)
                a = torch.argmax(a_probs).unsqueeze(0)
                log_probs_controller.append(torch.log(a_probs[a]))
                a = torch.tensor(a).float()
                x_imagined, r = imagination(x_real, a)
                x_imagined = torch.tensor(x_imagined).float()
                r = torch.tensor([r]).float()
                tot_rewards.append(r)
                n_imagined += 1
                tot_imagination_steps += 1

            elif u == 2:
                a_probs = controller(x_real, history)
                a = torch.argmax(a_probs).unsqueeze(0)
                log_probs_controller.append(torch.log(a_probs[a]))
                a = torch.tensor(a).float()
                x_imagined, r = imagination(x_imagined, a)
                x_imagined = torch.tensor(x_imagined).float()
                r = torch.tensor([r]).float()
                tot_rewards.append(r)
                n_imagined += 1
                tot_imagination_steps += 1

            d = torch.cat([u, a, r, x_real, x_imagined, torch.tensor([n_real]), torch.tensor([n_imagined])]).unsqueeze(0)
            history = memory(d)
        
            if done or truncated:
                print(f"Ended episode {ep+1}\n")
                break

        #imagination loss
        imagined_states = torch.stack([s for s in imagined_states], dim=0)
        real_states = torch.stack([s.type(torch.FloatTensor) for s in real_states], dim=0)
        imagined_rewards = torch.stack([s for s in imagined_rewards], dim=0)
        real_rewards = torch.stack([s for s in real_rewards], dim=0)
        imagination_loss = F.mse_loss(imagined_states, real_states) + F.mse_loss(imagined_rewards, real_rewards)
        imagination_optim.zero_grad()
        imagination_loss.backward()
        imagination_optim.step()

        #manager loss
        external_loss_manager = 0
        discounts = torch.pow(gamma, torch.arange(len(real_rewards)))

        for t in range(len(real_rewards)):
            G = (discounts[:len(real_rewards)-t]*real_rewards[t:]).sum()
            external_loss_manager += -(gamma**t)*G*external_log_probs_manager[t]
        
        imagination_cost = 0.1
        internal_loss = imagination_cost*tot_imagination_steps #I consider the internal loss as a penalty fo imagining
        
        manager_loss = -(external_loss_manager + internal_loss)
        manager_optim.zero_grad()
        manager_loss.backward(retain_graph=True)
        manager_optim.step()

        #controller and memory loss
        con_mem_loss = 0
        discounts_tot = torch.pow(gamma, torch.arange(len(tot_rewards)))
        tot_rewards = torch.stack([s for s in tot_rewards], dim=0)
        for t in range(len(tot_rewards)):
            G = (discounts_tot[:len(tot_rewards)-t]*tot_rewards[t:]).sum()
            con_mem_loss += -(gamma**t)*G*log_probs_controller[t]
        con_mem_optim.zero_grad()
        con_mem_loss.backward()
        con_mem_optim.step()

        

if __name__ == '__main__':
    episodes = 10
    n_max_real_steps = 100
    n_max_imagined_steps = 5
    gamma = 0.9

    train(episodes, n_max_real_steps, n_max_imagined_steps, gamma)