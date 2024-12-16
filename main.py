from models import Manager, Controller, Imagination, Memory
import gymnasium as gym
import torch

def train(episodes, env, n_max_real_steps, n_max_imagined_steps):
    state_size = len(env.observation_space)
    action_size = env.action_space.n
    history_size = 16 #TODO

    manager = Manager(state_size, history_size)
    controller = Controller(state_size, history_size, action_size)
    imagination = Imagination(state_size, action_size)
    memory = Memory(history_size, history_size)#TODO

    for ep in range(episodes):
        s, _ = env.reset()

        history = torch.zeros(history_size)#TODO
        n_real, n_imagined = 0, 0
        x_real, x_imagined = s, s

        while n_real < n_max_real_steps:
            u = manager(x_real, history)

            if u == 0 or n_imagined > n_max_imagined_steps:

                a = controller(x_real, history)
                x_real, r, done, truncated, _ = env.step(a)

                n_real += 1
                n_imagined = 0
                x_imagined = x_real

            if u == 1:
                a = controller(x_real, history)
                x_imagined, r = imagination(x_real, a)
                n_imagined += 1

            if u == 2:
                a = controller(x_imagined, history)
                x_imagined = imagination(x_imagined, a)
                n_imagined += 1

            history = memory(history)#TODO
            if done or truncated:
                break


if __name__ == '__main__':
    episodes = 10
    n_max_real_steps = 5
    n_max_imagined_steps = 5

    env = gym.make('Blackjack-v1', natural=False, sab=False, render_mode='human')
    train(episodes, env, n_max_real_steps, n_max_imagined_steps)