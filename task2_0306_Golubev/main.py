from ppo_agent import PPOAgent
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    ppo_agent = PPOAgent()

    if args.train:
        ppo_agent.learn()
    else:
        ppo_agent.test()