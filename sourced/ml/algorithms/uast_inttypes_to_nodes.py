from typing import Union, Iterable, Tuple

from bblfsh import Node
import numpy

from sourced.ml.algorithms.uast_to_bag import Uast2BagThroughSingleScan


class Uast2QuantizedChildren(Uast2BagThroughSingleScan):
    """
    Converts a UAST to a bag of children counts.
    """

    def __init__(self, npartitions: int=10):
        self.npartitions = npartitions
        self.levels = {}

    def node2key(self, node: Node) -> Union[str, Tuple[str, int]]:
        """
        :param node: a node in UAST.
        :return: The string which consists of the internal type of the node and its number of
        children.
        """
        if not self.levels:
            return node.internal_type, len(node.children)
        qm = self.levels[node.internal_type]
        quant_index = numpy.searchsorted(qm, len(node.children), side="right") - 1
        return "%s_%d" % (node.internal_type, quant_index)

    def quantize(self, frequencies: Iterable[Tuple[str, Iterable[Tuple[int, int]]]]):
        for key, vals in frequencies:
            self.levels[key] = self.quantize_unwrapped(vals)

    def quantize_unwrapped(self, children_freq: Iterable[Tuple[int, int]]) -> numpy.ndarray:
        """
        Builds the quantization partition P that is a vector of length nb_partitions \
        whose entries are in strictly ascending order.
        Quantization of x is defined as:
            0 if x <= P[0]
            m if P[m-1] < x <= P[m]
            n if P[n] <= x

        :param children_freq: distribution of the number of children.
        :return: The array with quantization levels.
        """
        levels = numpy.zeros(self.npartitions + 1, dtype=numpy.int32)
        children_freq = sorted(children_freq)
        max_nodes_per_bin = sum(i[1] for i in children_freq) / self.npartitions
        levels[0] = children_freq[0][0]
        accum = children_freq[0][1]
        i = 1
        for v, f in children_freq[1:]:
            accum += f
            if accum > max_nodes_per_bin:
                accum = f
                if i < len(levels):
                    levels[i] = v
                    i += 1
        last = children_freq[-1][0]
        if i < len(levels):
            levels[i:] = last
        else:
            levels[-1] = last
        return levels
