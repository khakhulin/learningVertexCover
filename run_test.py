from collections import namedtuple

import torch

from args import get_args
from env import VertexCover
import torch.nn.functional as F
from agent import Agent
import time
from copy import deepcopy as dc
import networkx as nx
import numpy as np
import tqdm
#import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx, from_networkx

args = get_args()
Result = namedtuple("Result", "score vertices")
PrintGraph = namedtuple("PrintGraph", "graph pos mvc not_mvc")


def print_graph(g, node_size=100):
    nx.draw_networkx_nodes(g.graph, g.pos, node_size=node_size, nodelist=g.mvc, node_color="tab:orange")
    nx.draw_networkx_nodes(g.graph, g.pos, node_size=node_size, nodelist=g.not_mvc, node_color="tab:blue")
    nx.draw_networkx_edges(g.graph, g.pos, alpha=0.7, width=3)
    ax = plt.gca()
    ax.margins(0.11)
    plt.tight_layout()
    plt.axis("off")
    plt.show()


def test_problem(args, agent):
    problem = VertexCover(num_nodes, p_edge)
    _G, actions, rewards, retime = run_agent(problem, agent)
    real_score, agent_score = len(problem.mvc), len(actions)
    heuristic_result = real_score
    agent_result = agent_score
    return _G, heuristic_result, agent_result, retime


def run_agent(env, agent):
    init_state, done = env.reset()
    state = dc(init_state)

    _G = to_networkx(init_state.g)
    T1 = time.time()
    [idx1, idx2] = env.get_permitted_actions(state)
    actions = []
    rewards = []
    worst_cost = 0.0

    sum_r = 0.0
    while not done:
        G = state.g.to(device)
        pi, val = agent(G)
        pi = pi.squeeze()
        pi[idx1] = -float('Inf')
        pi = F.softmax(pi, dim=0)
        dist = torch.distributions.categorical.Categorical(pi)
        action = dist.sample()
        new_state, reward, done = env.step(state, action)
        [idx1, idx2] = env.get_permitted_actions(new_state)
        state = new_state
        sum_r += -reward.item()
        worst_cost = np.maximum(worst_cost, reward.item())
        if done:
            reward = -torch.tensor(worst_cost)

        rewards.append(reward.item())
        actions.append(action.item())
    T2 = time.time()
    return _G, actions, rewards, T2 - T1


cuda_flag = torch.cuda.is_available()
device = torch.device('cuda' if cuda_flag else 'cpu')
num_nodes = args.n
p_edge = args.p
draw_graph = args.verbose

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
agent_mvc = Agent(2, args.hidden_dim, device=device).to(device)
agent_mvc.load_state_dict(torch.load(f'{args.exp_name}.pth'))

heuristic_result_scores = []
agent_result_scores = []

for num_nodes in range(args.min_n, args.max_n):
    heuristic_result_scores_n = []
    agent_result_scores_n = []
    for i in tqdm.tqdm(range(5)):
        env = VertexCover(num_nodes, p_edge, args.random_graph_type)
        _G, heuristic_result, agent_result, res_time = test_problem(env, agent_mvc)

        heuristic_result_scores_n.append(heuristic_result)
        agent_result_scores_n.append(agent_result)
    
    heuristic_result_scores.append(heuristic_result_scores_n)
    agent_result_scores.append(agent_result_scores_n)

    heuristic_result_scores_n = np.asarray(heuristic_result_scores_n)
    agent_result_scores_n = np.asarray(agent_result_scores_n)
    print(f'Num nodes: {num_nodes}, heuristic score: {heuristic_result_scores_n.mean()}, agent score {agent_result_scores_n.mean()}')

heuristic_result_scores = np.asarray(heuristic_result_scores)
agent_result_scores = np.asarray(agent_result_scores)

np.save(f'logs/{args.exp_name}_heuristic.npy', heuristic_result_scores)
np.save(f'logs/{args.exp_name}_agent.npy', agent_result_scores)