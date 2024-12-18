from models import Manager, Controller, Imagination, Memory
import gymnasium as gym
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
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

    manager_losses, imagination_losses, con_mem_losses = [], [], []
    plot_rewards = []

    for ep in tqdm(range(episodes)):
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

        #keep track of losses
        manager_losses.append(manager_loss.item())
        imagination_losses.append(imagination_loss.item())
        con_mem_losses.append(con_mem_loss.item())

        plot_rewards.append(real_rewards.sum())

    plt.figure(figsize=(10, 5))
    plt.plot(manager_losses, label="Manager Loss")
    plt.plot(imagination_losses, label="Imagination Loss")
    plt.plot(con_mem_losses, label="Controller & Memory Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("Losses Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()


    window_size = 100
    rolling_avg_rewards = np.convolve(plot_rewards, np.ones(window_size)/window_size, mode='valid')

    plt.figure(figsize=(10, 5))
    plt.plot(plot_rewards, label="Episode Rewards", alpha=0.5)
    plt.plot(rolling_avg_rewards, label=f"Rolling Average (Window={window_size})", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Rolling Average of Rewards")
    plt.legend()
    plt.grid(True)
    plt.show()

    return manager, controller, imagination, memory



def evaluate(manager, controller, imagination, memory, eval_episodes=10000):
    env = gym.make('Blackjack-v1', natural=False, sab=False)
    
    manager.eval()
    controller.eval()
    imagination.eval()
    memory.eval()

    tot_rewards = []
    win_count, loss_count, draw_count = 0, 0, 0

    state_size = len(env.observation_space)

    history_size = state_size*2 + 5  #x_real, x_imagined, a, u, n_real, n_imagined, reward


    for ep in range(eval_episodes):
        s, _ = env.reset()
        s = torch.tensor(s).float()
        done, truncated = False, False
        tot_reward = 0

        history = torch.zeros(history_size)

        x_real = s
        n_real, n_imagined = 0, 0

        while not(done or truncated):
            with torch.no_grad():
                u_probs = manager(x_real, history)
                u = torch.argmax(u_probs).unsqueeze(0)

                if u == 0:
                    a_probs = controller(x_real, history)
                    a = torch.argmax(a_probs).unsqueeze(0)
                    x_real, r, done, truncated, _ = env.step(int(a.item()))
                    x_real = torch.tensor(x_real).float()
                    tot_reward += r
                    n_real += 1
                    n_imagined = 0


                elif u in {1, 2}:
                    a_probs = controller(x_real, history)
                    a = torch.argmax(a_probs).unsqueeze(0)
                    x_real, _ = imagination(x_real, torch.tensor([a.item()]).float())
                    n_imagined += 1

                # Update history
                d = torch.cat([x_real, x_real, torch.tensor([a.item()]), u, torch.tensor([r]), torch.tensor([n_real]), torch.tensor([n_imagined])]).unsqueeze(0)
                history = memory(d)

        tot_rewards.append(tot_reward)
        if tot_reward > 0:
            win_count += 1
        elif tot_reward < 0:
            loss_count += 1
        else:
            draw_count += 1

    avg_reward = np.mean(tot_rewards)
    win_rate = win_count / eval_episodes
    loss_rate = loss_count / eval_episodes
    draw_rate = draw_count / eval_episodes

    print(f"Evaluation Results over {eval_episodes} episodes:")
    print(f"  Average Reward: {avg_reward:.2f}")
    print(f"  Win Rate: {win_rate * 100:.2f}%")
    print(f"  Loss Rate: {loss_rate * 100:.2f}%")
    print(f"  Draw Rate: {draw_rate * 100:.2f}%")

    return avg_reward, win_rate, loss_rate, draw_rate



if __name__ == '__main__':
    episodes = 50000
    n_max_real_steps = 1000
    n_max_imagined_steps = 4
    gamma = 0.9

    manager, controller, imagination, memory = train(episodes, n_max_real_steps, n_max_imagined_steps, gamma)
    avg_reward, win_rate, loss_rate, draw_rate = evaluate(manager, controller, imagination, memory)