import gymnasium as gym
from gymnasium.wrappers import RecordVideo

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


#Определение сети политики (actor) для дискретного пространства действий
class Actor(nn.Module):
    def __init__(self, stateDim, actionDim, hiddenSize=64):
        super(Actor, self).__init__()

        self.log_std = nn.Parameter(torch.zeros(actionDim)) #self.log_std = nn.Parameter(torch.zeros(1, actionDim))

        self.net = nn.Sequential(
            nn.Linear(stateDim, hiddenSize),
            nn.ReLU(),  #nn.Tanh(),
            nn.Linear(hiddenSize, hiddenSize),
            nn.ReLU(),  #nn.Tanh(),
            nn.Linear(hiddenSize, actionDim),
            ##1) Вариант с тангенсом
            #nn.Tanh() - был нужен для обрезания значений сейчас
            ##2) Вариант без -- ничего не нужно
        )

    def forward(self, x):
        logits = self.net(x) #предсказание сети актора
        return logits

    def get_dist(self, state):
        logits = self.forward(state)
        return Normal(loc=logits, scale=torch.exp(self.log_std))

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist = self.get_dist(state)
        ##1) Вариант с тангенсом
        # action = dist.sample()
        # return  action.item(), dist.log_prob(action).item()
        ##2)  Вариант без
        action = dist.sample()
        clamp = torch.clamp(action, -1.0, 1.0)
        return clamp.item(), dist.log_prob(clamp).item()

# Определение value-сети (critic)
class Critic(nn.Module):
    def __init__(self, stateDim, hiddenSize=64):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(stateDim, hiddenSize),
            nn.Tanh(),
            nn.Linear(hiddenSize, hiddenSize),
            nn.Tanh(),
            nn.Linear(hiddenSize, 1)
        )
    def forward(self, state):
        return self.net(state)


#Получение траектории изменения состояний для одной итерации
def collect_trajectories(policy, numSteps):
    states = []
    actions = []
    rewards = []
    dones = []
    logProbs = []
    episodeRewards = []

    state, _ = env.reset()
    episodeReward = 0

    for _ in range(numSteps):
        action, logProb = policy.act(state)

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

def cra_func(rewards, dones, values, gamma):
    returns = []
    advantages = []
    R = 0
    for reward, done, value in zip(reversed(rewards), reversed(dones), reversed(values)):
        if done:
            R = 0
        R = reward + gaeLambda * gamma * R
        returns.insert(0, R)
        advantage = R - value
        advantages.insert(0, advantage)

    returns = np.array(returns)
    advantages = np.array(advantages)

    ##
    returns = (returns - returns.mean()) / (returns.std() + 1e-8) #MAE
    ##
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) #MAE
    return returns, advantages

######################################################################################################################


# Гиперпараметры PPO
numIterations = 1000 #Число глобальных проходов (numSteps * ppoEpochs)
numSteps = 2048 #количество шагов для сбора одного батча. Число данных рассматриваемых на одной эпохе
ppoEpochs = 10 # число эпох
miniBatchSize = 32 #64 # размер батча
gamma = 0.99 #0.5 #0.99 #коэффициент дисконтирования
gaeLambda = 0.95 #Украл со статей
clipRatio = 0.08 #0.2 # 0.01 #коэффициент обрезки PPO
valueCoef = 0.5 #0.6 #0.5 #коэф. value loss
entropyCoef = 0.4 #0.2 #0.01 #коэф. энтропийного бонуса
lr = 3e-4 #шаг обучения при градиентном спуске

#Параметры среды и устройства
env_name = 'MountainCarContinuous-v0'
env = gym.make(env_name, render_mode="rgb_array")

env = RecordVideo(
        env,
        video_folder="./videos",
        fps=9,
        video_length=999, #Максимальное число шагов согласно среде
        episode_trigger=lambda t: t % 1 == 0
    )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

stateDim = env.observation_space.shape[0]
actionDim = env.action_space.shape[0]

actor = Actor(stateDim, actionDim).to(device)
actorOptimizer = optim.Adam(actor.parameters(), lr=lr)

critic = Critic(stateDim).to(device)
criticOptimizer = optim.Adam(critic.parameters(), lr=lr)

#Цикл обучения
for iteration in range(numIterations):
    iterationData = collect_trajectories(actor, numSteps)

    states = torch.FloatTensor(iterationData['states']).to(device)
    actions = torch.FloatTensor(iterationData['actions']).to(device)
    oldLogProbs = torch.FloatTensor(iterationData['logProbs']).to(device)

    with torch.no_grad():
        values = critic(states).squeeze().cpu().numpy()

    returns, advantages = cra_func(iterationData['rewards'], iterationData['dones'], values, gamma)

    returns = torch.FloatTensor(returns).to(device)
    advantages = torch.FloatTensor(advantages).to(device)

    datasetSize = states.size(0)
    indices = np.arange(datasetSize)

    for epoch in range(ppoEpochs):
        np.random.shuffle(indices)

        for start in range(0, datasetSize, miniBatchSize):
            end = start + miniBatchSize

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
                torch.clamp(ratio, 1 - clipRatio, 1 + clipRatio) * miniAdvantages
            )

            # Расчёт ошибки
            actorLoss = -torch.min(surrogate1, surrogate2).mean()

            entropyLoss = dist.entropy().mean()

            valueEstimates = critic(miniStates.squeeze())

            criticLoss = (miniReturns - valueEstimates).pow(2).mean() # MSE

            loss =  actorLoss + valueCoef * criticLoss - entropyCoef * entropyLoss

            # Запуск оптимизаторов
            actorOptimizer.zero_grad()
            criticOptimizer.zero_grad()

            loss.backward()

            actorOptimizer.step()
            criticOptimizer.step()

    avgReward = np.mean(iterationData['episodeRewards'])

    print(f'Iter: {iteration}, Loss: {loss.item():.4f}, AvgReward: {avgReward}')

    if avgReward >= -30: #-20 #Получено опытным путём. !Для тангенса -60
        print('Task is completed')
        break

