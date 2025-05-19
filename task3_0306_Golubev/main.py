from Agent import Agent
import argparse


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--train', help='Training mode', action='store_true')
	args = parser.parse_args()

	if not args.train:
		Agent.test()
	else:
		print('Input Ctrl-C to stop the training')
		Agent.train()