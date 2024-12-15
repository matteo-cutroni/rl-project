import gymnasium as gym

def train(episodes, env):
    total_reward = 0

    for ep in range(episodes):
        s, _ = env.reset()
        for t in range(100):
            s1, r, done, truncated, info = env.step(0)
            total_reward += r
            print(total_reward)
            if done or truncated:
                break


if __name__ == '__main__':
    episodes = 10
    env = gym.make('Blackjack-v1', natural=False, sab=False, render_mode='human')
    train(episodes, env)