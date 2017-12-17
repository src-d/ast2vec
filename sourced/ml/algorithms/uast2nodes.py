from collections import defaultdict
import numpy

from sourced.ml.algorithms.uast_ids_to_bag import FakeVocabulary


class UastNode2Bag():
    """
    Converts a UAST to a bag of features that are node specific.
    The features can be either (internal types, number of children)
    or (roles, number of children)
    """
    def __init__(self, type2ind=None, roles2ind=None, children2ind=None):
        """
        :param type2ind: The mapping from internal types to bag of keys. \
            If None, no mapping is performed.
        :param roles2ind: The mapping from roles to bag of keys.
        :param children2ind: The mapping from the number of children to bag of keys. \
            :class:'Quantization' is used if it is not specified.
        """
        self.type2ind = FakeVocabulary() if type2ind is None else type2ind
        self.roles2ind = FakeVocabulary() if roles2ind is None else roles2ind
        self.children2ind = Quantization() if children2ind is None else children2ind

    def _uast2nodes(self, root):
        """
        :param uast: The UAST root node.
        :return: The list of the nodes composing the UAST.
        """
        nodes = []
        queue = [root]
        while queue:
            child = queue.pop(0)
            queue.extend(child.children)
            nodes.append(child)
        return nodes

    def _nodes2children(self, nodes):
        """
        :param nodes: List of all nodes from a UAST.
        :return: The list of the number of children of all these nodes.
        """
        nb_children = []
        for n in nodes:
            nb_children.append(len(n.children))
        return nb_children

    def uast_to_bag(self, uast, feature):
        """
        Converts a UAST to a bag-of-features. The weights are feature frequencies.
        The number of children are preprocessed by :class:`Quantization`

        :param uast: The UAST root node.
        :param feature: The first feature we want to extract from the nodes of the UAST.
        :return:
        """
        bag = defaultdict(int)
        nodes = self._uast2nodes(uast)
        nb_children = self._nodes2children(nodes)
        self.children2ind.build_partition(nb_children)
        for n in nodes:
            bag[self.node2ind(n, feature)] += 1
        return bag

    def node2ind(self, node, feature):
        """
        str format is required for wmhash.Bags.Extractor
        :param node: a node of UAST
        :param feature: The first feature we want to extract from the nodes of the UAST. \
        This first feature should be either "internal_type" or "roles", \
        the second being the number of children anyway.
        :return:
        """
        if feature is "internal_type":
            return " ".join([self.type2ind[node.internal_type],
                             str(self.children2ind.process_value(len(node.children)))])
        elif feature is "roles":
            return " ".join([str(self.roles2ind[node.roles]),
                             str(self.children2ind.process_value(len(node.children)))])
        else:
            raise NotImplementedError


class Quantization():
    """
    Algorithm that performs the quantization of a list of values.
    """
    def __init__(self, fineness=0.005):
        """
        :param fineness: The number of partitions grows with the fineness of the quantization. \
        The fineness parameter must be included between 0 excluded and 1.
        """
        self.fineness = fineness
        self.partition = []

    def build_partition(self, values):
        """
        Builds the partition of the quantization.
        It is a list of increasing integers equally partitioning the distribution of values.
        The length of the partition list increase with the fineness.

        :param values: List of values we want to quantize
        :return:
        """
        values.sort()
        max_nodes_per_bin = self.fineness * len(values)
        unique_values = list(set(values))
        max_values = max(values)
        ind = 0
        while True:
            nb_nodes_in_bin = 0
            while (nb_nodes_in_bin < max_nodes_per_bin) and (unique_values[ind] < max_values):
                nb_nodes_in_bin += len([x for x in values if x == unique_values[ind]])
                ind += 1
            if unique_values[ind] != max_values:
                self.partition.append(unique_values[ind - 1])
            else:
                self.partition.append(unique_values[ind] + 1)
                break

    def process_value(self, value):
        """
        Processes value according to the quantization algorithm. \
        Behaves like a stair function whose set of permissible outputs are the values in partition.

        :param value: value we want to quantize.
        :return:
        """
        linear_values = numpy.arange(len(self.partition))
        idx = numpy.searchsorted(self.partition, value, side="right")
        return linear_values[idx-1]
