import collections
import pdb

from torch import nn
import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy as dc


class ActorCritic(nn.Module):
    def __init__(self,
                 problem,
                 agent,
                 gamma=0.98,
                 num_episodes=1,
                 use_discount_reward=True,
                 device=torch.device('cpu')):
        super(ActorCritic, self).__init__()
        self.problem = problem
        self.model = agent
        self.gamma = gamma
        self.num_episodes = num_episodes
        self.device = device
        self.log_reward = False
        self.discount_reward = use_discount_reward
        #self.entropy_weight = 0.00009
        self.entropy_weight = 0.001
        self.entropy_factor = 0.95
        self.default_log = collections.defaultdict(list)

    def _add_to_torch_arr(self, tensor, val, i):
        if i == 0:
            tensor = val
        else:
            tensor = torch.cat([tensor, val], dim=0)
        return tensor

    def run_episode(self, use_log=True, reset_fix=False):
        sum_r = 0
        if reset_fix:
            state, done = self.problem.reset_fixed()
        else:
            state, done = self.problem.reset()
        closed_indexes, _ = self.problem.get_permitted_actions(state)
        t = 0
        worst_cost = 0.0
        real_size = state.g.x.shape[0]
        policy_proba, reward_episode, value_episode = None, None, None
        while not done:
            G = dc(state.g).to(self.device)
            G.x = G.x.to(self.device)
            proba, val = self.model(G)
            proba = proba.squeeze()
            proba[closed_indexes] = -float('Inf')
            proba = F.softmax(proba, dim=0)
            dist = torch.distributions.categorical.Categorical(proba)
            action = dist.sample()

            new_state, reward, done = self.problem.step(dc(state), action)
            closed_indexes, _ = self.problem.get_permitted_actions(new_state)
            state = dc(new_state)
            sum_r += reward
            if use_log:
                worst_cost = np.maximum(worst_cost, -reward.item())
                if self.log_reward:
                    reward = -torch.log(1 + reward)
                # reward = torch.tensor(reward, dtype=torch.float)
                # if done:
                #     reward = torch.tensor(sum(reward_episode))
                #     last_reward = torch.tensor(sum(reward_episode))
            policy_proba = self._add_to_torch_arr(policy_proba, proba[action].unsqueeze(0), t)
            reward_episode = self._add_to_torch_arr(reward_episode, reward.unsqueeze(0), t)
            value_episode = self._add_to_torch_arr(value_episode, val.unsqueeze(0), t)
            t += 1

        tot_return = reward_episode[-1].item()
        if self.discount_reward:
            for i in range(reward_episode.shape[0] - 1):
                reward_episode[-2 - i] = reward_episode[-2 - i] + self.gamma * reward_episode[-1 - i]

        return policy_proba, reward_episode, value_episode, tot_return, len(self.problem.mvc), real_size

    def update_model(self, proba, reward, value):
        advantage = reward.squeeze() - value.squeeze().detach()
        policy_loss = -(torch.log(proba) * advantage).mean()
        value_loss = F.smooth_l1_loss(value.squeeze().float(), reward.squeeze().float())
        entropy = -(proba * proba.log()).mean()
        loss = policy_loss + value_loss - self.entropy_weight * entropy

        self.entropy_weight *= self.entropy_factor

        dict_log = {'TD_error': value_loss,
                    'entropy': entropy,
                    'policy_loss': policy_loss,
                    'cost': reward.squeeze()[-1],
                    }
        for k, v in dict_log.items():
            self.default_log[k].append(v.cpu().detach().item())

        return loss

    def forward(self, reset_fix: bool=False):
        mean_return = 0
        mean_rcost = 0.0
        mean_number_nodes = 0.0
        policy_proba, reward_, V = None, None, None
        for i in range(self.num_episodes):
            pi, r, v, tot_return, _real_cost, r_gsize = self.run_episode(reset_fix=reset_fix)
            mean_return = mean_return + tot_return
            mean_rcost = mean_rcost + _real_cost
            mean_number_nodes = r_gsize + mean_number_nodes
            policy_proba = self._add_to_torch_arr(policy_proba, pi, i)
            reward_ = self._add_to_torch_arr(reward_, r, i)
            V = self._add_to_torch_arr(V, v, i)
            # print(reward_, r_gsize, len(reward_), sum(reward_))
            # exit()
        reward_ = reward_.to(self.device)
        mean_return = mean_return / self.num_episodes
        mean_rcost = mean_rcost / self.num_episodes
        mean_number_nodes = mean_number_nodes / self.num_episodes
        loss = self.update_model(policy_proba, reward_, V)
        self.default_log['train_reward'].append(mean_return)
        self.default_log['real_cost'].append(mean_rcost)
        
        return loss