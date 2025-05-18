from nets import *
import gymnasium as gym
import matplotlib.pyplot as plt
import time
import numpy as np
import torch.optim as optim


class PPOAgent:
    def __init__(self):
        self.iterations = 1000
        self.steps = 1024
        self.epochs = 10
        self.mini_batch_size = 32
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_ratio = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.4
        self.lr = 3e-4
        self.test_episodes = 10
    

    def __collect_trajectories(self, env, policy):
        states = []
        actions = []
        rewards = []
        dones = []
        logProbs = []
        episodeRewards = []

        state, _ = env.reset()
        episodeReward = 0

        for _ in range(self.steps):
            action, logProb = policy.commit_action(state)

            nextState, reward, terminated, truncated, _ = env.step(np.array([action]))
            done = terminated or truncated
            states.append(state)
            state = nextState

            rewards.append(reward)
            actions.append(action)
            logProbs.append(logProb)
            dones.append(done)

            episodeReward += reward

            if done:
                state, _ = env.reset()
                episodeRewards.append(episodeReward)
                episodeReward = 0
        return {
            'states': np.array(states),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'dones': np.array(dones),
            'logProbs': np.array(logProbs),
            'episodeRewards': np.array(episodeRewards)
        }


    def __returns_and_advantages(self, rewards, dones, values):
        returns = []
        advantages = []
        R = 0
        for reward, done, value in zip(reversed(rewards), reversed(dones), reversed(values)):
            if done:
                R = 0
            R = reward + self.gae_lambda * self.gamma * R
            returns.insert(0, R)
            advantage = R - value
            advantages.insert(0, advantage)

        returns = np.array(returns)
        advantages = np.array(advantages)

        # нормализация
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages
    

    def __save_graph(self, rewards):
        fig = plt.figure(1)

        plt.xlabel('Episode')
        plt.ylabel('Mean reward')
        plt.plot(rewards)
        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        fig.savefig('./graph.png')
        plt.close(fig)
    

    def learn(self):
        env = gym.make('MountainCarContinuous-v0', render_mode=None)

        actor = Actor(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
        actorOptimizer = optim.Adam(actor.parameters(), lr=self.lr)

        critic = Critic(env.observation_space.shape[0]).to(device)
        criticOptimizer = optim.Adam(critic.parameters(), lr=self.lr)

        mean_rewards = []

        for iteration in range(self.iterations):
            time_now = time.time()

            iterationData = self.__collect_trajectories(env, actor)

            states = torch.FloatTensor(iterationData['states']).to(device)
            actions = torch.FloatTensor(iterationData['actions']).to(device)
            oldLogProbs = torch.FloatTensor(iterationData['logProbs']).to(device)

            with torch.no_grad():
                values = critic(states).squeeze().cpu().numpy()

            returns, advantages = self.__returns_and_advantages(iterationData['rewards'], iterationData['dones'], values)

            returns = torch.FloatTensor(returns).to(device)
            advantages = torch.FloatTensor(advantages).to(device)

            datasetSize = states.size(0)
            indices = np.arange(datasetSize)

            for epoch in range(self.epochs):
                np.random.shuffle(indices)

                for start in range(0, datasetSize, self.mini_batch_size):
                    end = start + self.mini_batch_size

                    miniIndices = indices[start:end]

                    miniStates = states[miniIndices]
                    miniActions = actions[miniIndices]
                    miniOldLogProbs = oldLogProbs[miniIndices]
                    miniReturns = returns[miniIndices]
                    miniAdvantages = advantages[miniIndices]

                    dist = actor.get_dist(miniStates)
                    newLogProbs = dist.log_prob(miniActions)

                    ratio = torch.exp(newLogProbs - miniOldLogProbs)

                    surrogate1 = (ratio * miniAdvantages)

                    surrogate2 = (
                        torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * miniAdvantages
                    )

                    # Расчёт ошибки
                    actorLoss = -torch.min(surrogate1, surrogate2).mean()
                    entropyLoss = dist.entropy().mean()
                    valueEstimates = critic(miniStates.squeeze())
                    criticLoss = (miniReturns - valueEstimates).pow(2).mean()
                    loss =  actorLoss + self.value_coef * criticLoss - self.entropy_coef * entropyLoss

                    # Запуск оптимизаторов
                    actorOptimizer.zero_grad()
                    criticOptimizer.zero_grad()

                    loss.backward()

                    actorOptimizer.step()
                    criticOptimizer.step()

            mean_reward = np.mean(iterationData['episodeRewards'])
            mean_rewards.append(mean_reward)

            torch.save(
            {
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
            },
            "./ppo_mountain_car.pth")

            print(f'Iteration: {iteration} | Loss: {loss.item():.4f} | Mean reward: {mean_reward} | Time spent: {round(time.time() - time_now, 2)} s')

            if mean_reward >= -30:
                break

        self.__save_graph(mean_rewards)
    

    def test(self):
        env = gym.make('MountainCarContinuous-v0', render_mode="human")

        policy = Actor(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
        policy.load_state_dict(torch.load('ppo_mountain_car.pth')['actor'])
        policy.eval()

        for _ in range(self.test_episodes):
            state, _ = env.reset()

            done = False

            while not done:
                action, logProb = policy.commit_action(state)

                nextState, reward, terminated, truncated, _ = env.step(np.array([action]))
                done = terminated or truncated
                state = nextState

                if done:
                    break