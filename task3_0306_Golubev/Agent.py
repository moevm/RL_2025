import gymnasium as gym
import flappy_bird_gymnasium
import os
from utils import evaluate_policy
import torch
from SAC import SAC


class Agent:
    def train():
        env = gym.make("FlappyBird-v0", render_mode=None, use_lidar=False)
        eval_env = gym.make("FlappyBird-v0", use_lidar=False)
		
        max_train_steps = 400000
        save_interval = 10000
        eval_interval = 2000
        random_steps = 10000
        update_every = 50

        env_seed = 0		
        torch.manual_seed(env_seed)
        torch.cuda.manual_seed(env_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if not os.path.exists('model'):
            os.mkdir('model')	
        agent = SAC(env.observation_space.shape[0], env.action_space.n)
		
        total_steps = 0
        graph_points = []
		
        while total_steps < max_train_steps:
            s, info = env.reset(seed=env_seed)
            env_seed += 1
            done = False

            while not done:
                if total_steps < random_steps:
                    a = env.action_space.sample()
                else:
                    a = agent.select_action(s)
                s_next, r, dw, tr, info = env.step(a)
                done = (dw or tr)

                agent.replay_buffer.add(s, a, r, s_next, dw)
                s = s_next

                '''update if its time'''
                if total_steps >= random_steps and total_steps % update_every == 0:
                    for j in range(update_every):
                        agent.train()

                '''record & log'''
                if total_steps % eval_interval == 0:
                    score = evaluate_policy(eval_env, agent)
                    print(f'reward: {score} | alpha: {agent.alpha}')
                    graph_points.append([total_steps, score])
                total_steps += 1

                '''save model'''
                if total_steps % save_interval == 0:
                    agent.save()
                    agent.save_graph(graph_points)
                    print('saved model')
		
        env.close()
        eval_env.close()
	

    def test():
        env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=False)
        env_seed = 0
		
        torch.manual_seed(env_seed)
        torch.cuda.manual_seed(env_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
		
        agent = SAC(env.observation_space.shape[0], env.action_space.n)		
        agent.load()

        test_episodes = 10
		
        for i in range(test_episodes):
            _ = evaluate_policy(env, agent)