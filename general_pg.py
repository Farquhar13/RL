import sys
sys.path.append('../')

import gym
from PolicyGradient import PolicyGradient
from ReplayMemory import Memory
import numpy as np
from torch.distributions.categorical import Categorical
import torch

"""
Testing with Reward-to-Go to verify things are working.
"""
class PG(PolicyGradient):
    def get_batch(self):
        # Get batch of experiences
        observations = [m[0] for m in self.memory]
        actions = [m[1] for m in self.memory]
        rewards = [m[3] for m in self.memory]
        episode_is_done_list = [m[4] for m in self.memory]

        episode_rewards = self.partition_by_episode(rewards, episode_is_done_list)

        return observations, actions, rewards, episode_is_done_list

    def compute_rewards_to_go(self, rewards):
        total_reward = sum(rewards)
        reward_to_go = np.zeros_like(rewards)
        for i in range(len(rewards)):
            if i != 0:
                total_reward -= rewards[i-1]
            reward_to_go[i] = total_reward

        return reward_to_go

    def critic(self):
        """ Return the value-estimation for each training example as a np.ndarray """

        _, _, rewards, episode_is_done_list = self.get_batch()

        # Compute rewards-to-go
        episode_rewards = self.partition_by_episode(rewards, episode_is_done_list)
        rewards_to_go = []
        for ep_rewards in episode_rewards:
            rewards_to_go += list(self.compute_rewards_to_go(ep_rewards))

        return rewards_to_go

    def get_policy(self, observation):
        """ Returns the output of calling the distribution """
        logits = self.policy_net(observation)
        return self.distribution(logits=logits)

    def choose_action(self, observation):
        """ Return the agent's action for the next time-step """
        if isinstance(observation, (np.ndarray, list)):
            observation = torch.tensor(observation, dtype=torch.float32)

        return self.get_policy(observation).sample().item() # item() to return scalar from torch.tensor

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=50) # 50
    parser.add_argument('--batch_size', type=int, default=5000) # 5000
    args = parser.parse_args()

    n_epochs = args.n_epochs
    batch_size = args.batch_size

    # Create gym environement
    env = gym.make('CartPole-v0')
    n_observations = env.observation_space.shape[0]
    n_actions =  env.action_space.n

    # Create model or agent
    agent = PG(n_observations, n_actions,)

    for e in range(n_epochs):
        # reset episode-specific and epoch-specific variables
        observation = env.reset()
        done = False
        score = 0
        time_step = 0
        episode_rewards = []
        episodes_lengths = []
        returns_per_episode = []

        while True:
            time_step += 1
            action = agent.choose_action(observation)
            #env.render() # Uncomment for video
            next_observation, reward, done, info = env.step(action)
            score += reward
            episode_rewards.append(reward)
            agent.remember(observation, action, next_observation, reward, done)
            observation = next_observation

            if done == True:
                #print("Episode {} out of {}, score: {}, length: {}".format(e+1, n_epsiodes, score, t+1))
                returns_per_episode.append(score)
                episodes_lengths.append(time_step)
                # reset episode-specific variables
                observation, done, score, time_step, episode_rewards = env.reset(), False, 0, 0, []

                # If enough experience collected, stop collecting experience, end epoch, and train
                if len(agent.memory) >= batch_size:
                    break

        loss = agent.learn()
        print('epoch: %3d \t loss: %.3f \t avg_ep_return: %.3f \t avg_ep_length: %.3f' %
        (e+1, loss, np.mean(returns_per_episode), np.mean(episodes_lengths)))
