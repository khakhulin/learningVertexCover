__all__ = ['VertexCover']

import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.utils import to_networkx, to_undirected, from_networkx, degree
from networkx.algorithms.approximation import min_weighted_vertex_cover

from graph import Graph


class State:
    def __init__(self, graph, visited_node=None, **kwargs):
        self.__dict__.update(kwargs)
        self.N = graph.num_nodes
        self.g = graph
        # self.nxg = to_networkx(graph)
        self.visited_node = visited_node


class BasicEnv:
    def __init__(self, N, prob=0.3, *args):
        self.N = N
        self.prob = prob
        self.initial_state = self._init_state(self.N, self.prob, *args)
        self.fix_graph = False

    @staticmethod
    def _init_state(N, prob, graph_type='erdos_renyi',
                     min_n=None, max_n=None, one_hot=False, g=None):
        """
        :param N:
        :param prob:
        :param graph_type: str
        :param one_hot: bool
        :param min_n: int
        :param max_n: int
        :return: State(random graph, visited nodes)
        """
        if g is None:
            g = Graph(graph_type, N, p=prob, min_n=min_n, max_n=max_n)
        gp = from_networkx(g.g)
        max_degree = g.nodes_num()
        idxs = gp.edge_index
        outdeg = degree(idxs[0], gp.num_nodes, dtype=torch.long)
        indeg = degree(idxs[1], gp.num_nodes, dtype=torch.long).to(torch.float)
        if one_hot:
            deg = F.one_hot(indeg, num_classes=max_degree + 1).to(torch.float)
        else:
            deg = indeg.unsqueeze(1) / indeg.max()
        gp['x'] = torch.cat((torch.zeros((max_degree, 1), dtype=torch.float), deg), dim=1)
        visited = torch.zeros(max_degree)
        deg = torch_geometric.utils.degree(gp.edge_index[0], gp.num_nodes)

        return State(gp, visited_node=visited,
                     degree=deg, org=g, graph_type=g.graph_type,
                     min_n=min_n, max_n=max_n)

    def reset(self):
        """
        Reset state
        :return: state, done
        """
        state = self._init_state(self.N, self.prob, graph_type=self.initial_state.graph_type,
                                 min_n=self.initial_state.min_n, max_n=self.initial_state.max_n)
        return state, False

    def reset_fixed(self):
        done = False
        return self.initial_state, done


class VertexCover(BasicEnv):
    def __init__(self, *args):
        super().__init__(*args)
        nxg = self.initial_state.org.get_graph()
        self.mvc = min_weighted_vertex_cover(nxg)

    def step(self, state, action):
        """
        :param state:
        :param action:
        :return: State, reward, done state
        """
        is_done = False
        state.g['x'][action, 0] = 1.0 - state.g['x'][action, 0]
        state.visited_node[action.item()] = 1.0
        reward = torch.tensor(-1., dtype=torch.float32)
        edge_visited = torch.cat((state.visited_node[state.g.edge_index[0]].unsqueeze(-1),
                                  state.visited_node[state.g.edge_index[1]].unsqueeze(-1)),
                                 dim=1).max(dim=1)[0]

        if edge_visited.mean().item() == 1.0:
            is_done = True

        return state, reward, is_done

    def get_permitted_actions(self, state):
        idx1 = (state.visited_node == 1.).nonzero()
        idx2 = (state.visited_node == 0.).nonzero()
        return idx1, idx2

    def reset(self):
        state = self._init_state(self.N, self.prob, graph_type=self.initial_state.graph_type,
                                 min_n=self.initial_state.min_n, max_n=self.initial_state.max_n)
        nxg = state.org.get_graph()
        self.mvc = min_weighted_vertex_cover(nxg)
        return state, False