import argparse


def get_args():
    parser = argparse.ArgumentParser()

    #  Experiment
    parser.add_argument('--cuda', default=False, action='store_true')
    parser.add_argument('--cuda_device', default=0, type=int)
    parser.add_argument('--seed', default=42, type=int, help='seed')

    parser.add_argument('--exp_name', default='default', type=str, help='name of the experiments')
    parser.add_argument('--resume_path', default=None, type=str, help='path to the weights of experiment')
    parser.add_argument('--problem', default='mvc', type=str, help='name of the experiments', choices=['mvc'])
    parser.add_argument('--random_graph_type', default='erdos_renyi', type=str, help='type of random graph generation')
    parser.add_argument('--reset_fix', default=False, action='store_true')

    parser.add_argument('--num_epochs', default=4, type=int, help='num of training epochs')
    parser.add_argument('--print_every', default=1, type=int, help='print train information')
    parser.add_argument('--save_agent_every', default=10, type=int, help='save agent every % epochs')

    parser.add_argument('--n', default=45, type=int, help='num of the nodes')

    parser.add_argument('--min_n', default=45, type=int, help='min num of the nodes')
    parser.add_argument('--max_n', default=45, type=int, help='max num of the nodes')

    parser.add_argument('--hidden-dim', default=128, type=int, help='hidden dimension')
    parser.add_argument('--p', default=0.15, type=float, help='probability of edge connectivity')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')

    args = parser.parse_args()

    return args