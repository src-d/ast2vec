import random
from collections import defaultdict, deque

from sourced.ml.algorithms.uast_ids_to_bag import FakeVocabulary


class UastStructure2BagBase:
    def uast_to_bag(self, uast):
        raise NotImplemented


class UastSeq2Bag(UastStructure2BagBase):
    """
    DFS traversal + preserves the order of node children.
    """
    def __init__(self, type2ind=None, stride=1, seq_len=5):
        self.type2ind = type2ind if type2ind is not None else FakeVocabulary()
        self.stride = stride
        self.seq_len = seq_len

    def _uast2seq(self, root, walk):
        nodes = defaultdict(deque)
        stack = [root]
        nodes[id(root)].extend(root.children)
        while stack:
            if nodes[id(stack[-1])]:
                child = nodes[id(stack[-1])].popleft()
                nodes[id(child)].extend(child.children)
                stack.append(child)
            else:
                walk.append(stack.pop())

    def uast_to_bag(self, uast):
        bag = defaultdict(int)
        seq = []
        self._uast2seq(uast, seq)

        seq = [self.node2ind(n) for n in seq]

        if isinstance(self.seq_len, int):
            seq_lens = [self.seq_len]
        else:
            seq_lens = self.seq_len
        for seq_ in seq:
            for seq_len in seq_lens:
                for i in range(0, len(seq_) - seq_len, self.stride):
                    # convert to str - requirement from wmhash.BagsExtractor
                    bag["".join(seq_[i:i + seq_len])] += 1
        return bag

    def node2ind(self, node):
        return self.type2ind[node.internal_type]


class Node:
    def __init__(self, parent=None, internal_type=None):
        self.parent = parent
        self.internal_type = internal_type
        self.children = []


class Uast2RandomWalks:
    """
    Generation of random walks for UAST.
    """

    def __init__(self, p_explore_neighborhood, q_leave_neighborhood, n_walks, n_steps,
                 type2ind=None):
        """
        :param p_explore_neighborhood: return parameter, p. Parameter p controls the likelihood of\
                                       immediately revisiting a node in the walk. Setting it to a\
                                       high value (> max(q, 1)) ensures that we are less likely to\
                                       sample an already visited node in the following two steps\
                                       (unless the next node in the walk had no other neighbor).\
                                       This strategy encourages moderate exploration and avoids\
                                       2-hop redundancy in sampling.
        :param q_leave_neighborhood: in-out parameter, q. Parameter q allows the search to\
                                     differentiate between “inward” and “outward” nodes. Such \
                                     walks obtain a local view of the underlying graph with \
                                     respect to the start node in the walk and approximate BFS \
                                     behavior in the sense that our samples comprise of nodes \
                                     within a small locality.
        :param n_walks: Number of walks from each node.
        :param n_steps: Number of steps in walk.
        """
        self.p_explore_neighborhood = p_explore_neighborhood
        self.q_leave_neighborhood = q_leave_neighborhood
        self.n_walks = n_walks
        self.n_steps = n_steps
        self.type2ind = type2ind if type2ind is not None else FakeVocabulary()

    def uast2walks(self, uast):
        starting_nodes = self.prepare_starting_nodes(uast)
        res = []
        for i in range(self.n_walks):
            for start_node in starting_nodes:
                res.append(self.random_walk(start_node))
        return res

    @staticmethod
    def _extract_node(node, parent):
        return Node(parent=parent, internal_type=node.internal_type)

    def prepare_starting_nodes(self, uast):
        starting_nodes = []
        self._prepare_starting_nodes(uast, None, starting_nodes)
        return starting_nodes

    def _prepare_starting_nodes(self, root, parent, starting_nodes):
        node = self._extract_node(node=root, parent=parent)
        starting_nodes.append(node)

        for ch in root.children:
            node.children.append(self._prepare_starting_nodes(
                ch, parent=node, starting_nodes=starting_nodes))

        return node

    def random_walk(self, node):
        walk = [node]
        while len(walk) < self.n_steps:
            walk.append(self.alias_sample(walk))

        walk = [self.node2feat(n) for n in walk]
        return walk

    def node2feat(self, node):
        return self.type2ind[node.internal_type]

    def alias_sample(self, walk):
        """
        Compare to node2vec this sampling is a bit simpler because there is no loop in tree ->
        so there are only 2 options with unnormalized probabilities 1/p & 1/q
        :param walk: list of visited nodes
        :return: next node to visit
        """
        # notation from article - t - node from previous step, v - last node
        v = walk[-1]

        if len(walk) == 1 and len(v.children) > 0:
            return random.choice(v.children)
        elif len(v.children) == 0:
            return v

        t = walk[-2]
        threshold = (1 / self.p_explore_neighborhood)
        threshold /= ((1 / self.p_explore_neighborhood) +
                      len(v.children) / self.q_leave_neighborhood)

        if random.random() <= threshold:
            return t

        options = []
        if v.parent is not None:
            options.append(v.parent)
        options.extend(v.children)
        return random.choice(options)


class UastRandomWalk2Bag(UastStructure2BagBase):
    def __init__(self, p_explore_neighborhood=0.5, q_leave_neighborhood=0.5, n_walks=5, n_steps=19,
                 stride=1, seq_len=(5, 6), seed=42):
        self.random_walker = Uast2RandomWalks(p_explore_neighborhood=p_explore_neighborhood,
                                              q_leave_neighborhood=q_leave_neighborhood,
                                              n_walks=n_walks, n_steps=n_steps)
        self.stride = stride

        if not isinstance(seq_len, (int, tuple, list)):
            raise TypeError("Unexpected type of seq_len: %s" % type(seq_len))

        self.seq_len = seq_len
        self.seed = seed

    def uast_to_bag(self, uast):
        if self.seed is not None:
            random.seed(self.seed)

        bag = defaultdict(int)
        walks = self.random_walker.uast2walks(uast)
        if isinstance(self.seq_len, int):
            seq_lens = [self.seq_len]
        else:
            seq_lens = self.seq_len
        for walk in walks:
            for seq_len in seq_lens:
                for i in range(0, len(walk) - seq_len, self.stride):
                    # convert to str - requirement from wmhash.BagsExtractor
                    bag["".join(walk[i:i + seq_len])] += 1
        return bag
