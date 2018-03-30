from bblfsh import Node
from collections import defaultdict
from typing import Dict


class Uast2BagBase:
    """
    Base class to convert UAST to a bag of anything.
    """
    def __call__(self, uast: Node):
        """
        Inheritors must implement this function.

        :param uast: The UAST root node.
        """
        raise NotImplemented


class Uast2BagThroughSingleScan(Uast2BagBase):
    """
    Constructs the bag by doing a single tree traversal and turning every node into a string.
    """
    def __call__(self, uast: Node) -> Dict[str, int]:
        result = defaultdict(int)
        stack = [uast]
        while stack:
            node = stack.pop()
            stack.extend(node.children)
            result[self.node2key(node)] += 1
        return result

    def node2key(self, node) -> str:
        raise NotImplemented
