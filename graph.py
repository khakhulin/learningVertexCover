"""
Classes for random graph generation
"""
import numpy as np
import networkx as nx

_graph_generation_method = {
    'erdos_renyi': nx.erdos_renyi_graph,
    'powerlaw': nx.powerlaw_cluster_graph,
    'barabasi_albert': nx.barabasi_albert_graph,
    'gnp_random_graph': nx.gnp_random_graph,
}


class Graph:
    def __init__(self, graph_type, N=None, min_n=None, max_n=None, p=0.1, m=4, seed=42, regenerate=True):
        if N is None and (min_n is None and max_n is None):
            raise ValueError('Declare correct Graph size')
        if N > 0:
            cur_n = N
        else:
            cur_n = np.random.randint(max_n - min_n + 1) + min_n
        # print(f'Create graph with {cur_n} number nodes')
        base_args = {'n': cur_n, 'p': p, 'seed': seed}
        if graph_type in ['barabasi_albert', 'powerlaw']:
            base_args['m'] = m  # Number of edges to attach from a new node to existing
            if graph_type == 'barabasi_albert':
                base_args.pop('p')
        self.g = _graph_generation_method[graph_type](**base_args)
        self.graph_type = graph_type
        while self.g.number_of_edges == 0:
            self.g = _graph_generation_method[graph_type](**base_args)

        self.adj_dense = nx.to_numpy_matrix(self.g, dtype=np.int32)

    def get_graph(self):
        return self.g

    def nodes_num(self):
        return nx.number_of_nodes(self.g)

    def edges(self):
        return self.g.edges()

    def neighbors(self, node):
        return nx.all_neighbors(self.g, node)

    def average_neighbor_degree(self, node):
        return nx.average_neighbor_degree(self.g, nodes=node)

    def adj(self):
        return self.adj_dense
