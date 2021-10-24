import os
import pickle
import torch
import numpy as np

from agent import Agent
from args import get_args
from actor_critic import ActorCritic
from env import VertexCover


def save_obj(obj, name):
    os.makedirs('logs', exist_ok=True)
    with open('logs/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)


def train_agent(args):
    agent = Agent(ndim=2, hidden_dim=args.hidden_dim)
    p_edge = args.p
    num_epochs = args.num_epochs
    print_every = args.print_every
    if args.min_n < args.max_n:
        # num_nodes = np.random.randint(args.max_n - args.min_n + 1) + args.min_n
        num_nodes = 0.0
    else:
        num_nodes = args.n
    problem = VertexCover(num_nodes, p_edge, args.random_graph_type, args.min_n, args.max_n)
    if args.resume_path is not None:
        weights = torch.load(args.resume_path)
        agent.model.load_state_dict(weights)

    algorithm = ActorCritic(problem=problem, agent=agent, )
    results_log = {'reward': [], 'value_loss': [], 'policy_loss': []}

    for i in range(num_epochs):
        logs = algorithm.train(args.reset_fix)
        r = logs.get('train_reward')
        v_loss = logs.get('TD_error')
        p_loss = logs.get('policy_loss')

        results_log['reward'].append(r[-1])
        results_log['value_loss'].append(v_loss[-1])
        results_log['policy_loss'].append(p_loss[-1])
        s_log = ""
        for k, it in results_log.items():
            s_log += f"|{k}  = {round(it[-1], 3)}|  "
        if i % print_every == 0:
            s_log += "cost " + str(-logs.get("train_reward", [0])[-1]) + "  | real cost " + str(
                logs.get('real_cost', [0])[-1])
            print(s_log)
    torch.save(agent.state_dict(), 'net.pth')
    save_obj(results_log, args.exp_name)


if __name__ == '__main__':
    args = get_args()
    train_agent(args)
