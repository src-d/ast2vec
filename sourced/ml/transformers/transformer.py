from io import StringIO

from sourced.ml.utils import PickleableLogger  # nopep8


class Transformer(PickleableLogger):
    BLOCKS = False  # set to True if __call__ launches PySpark

    def __init__(self, explain=None, **kwargs):
        super().__init__(**kwargs)
        self._children = []
        self._parent = None
        self._explained = explain

    @property
    def explained(self):
        if self._explained is None and self.parent is not None:
            return self.parent.explained
        return bool(self._explained)

    @property
    def children(self):
        return tuple(self._children)

    @property
    def parent(self):
        return self._parent

    def path(self):
        node = self
        path = []
        while node is not None:
            path.append(node)
            node = node.parent
        path.reverse()
        return path

    def link(self, transformer):
        self._children.append(transformer)
        transformer._parent = self
        return transformer

    def unlink(self, transformer):
        self._children.remove(transformer)
        transformer._parent = None
        return self

    def _explode(self, head, context):
        if context[-1] is not self:
            context.append(self)
            if not self._children or self.BLOCKS:
                self._log.info(self._format_pipeline(context))
            head = self(head)
        results = []
        for child in self._children:
            results.extend(child._explode(head, context.copy()))
        else:
            results.append(head)
        return results

    def explode(self, head=None):
        """
        Execute all the branches in the tree going from the node. The node itself
        is execute()-ed.
        :param head: The input to feed to the root. Can be None - the default one \
                     will be used if possible.
        :return: The results from all the leaves.
        """
        if self.parent is not None:
            head = self.execute(head)
        pipeline = [self]
        node = self
        while node.parent is not None:
            node = node.parent
            pipeline.append(node)
        pipeline.reverse()
        return self._explode(head, pipeline)

    def execute(self, head=None):
        """
        Execute the node together will all its dependencies, in order.
        :param head: The input to feed to the ultimate parent. Can be None - the default one \
                     will be used if possible.
        :return: The result of the execution.
        """
        pipeline = self.path()
        if not self._children:
            self._log.info(self._format_pipeline(pipeline))
        for node in pipeline:
            head = node(head)
        return head

    def graph(self, name="source-d", stream=None):
        if stream is None:
            stream = StringIO()
        stream.write("digraph %s {\n" % name)
        counters = {}
        nodes = {}
        queue = [(self, None)]
        while queue:
            node, parent = queue.pop(0)
            try:
                myself = nodes[node]
            except KeyError:
                index = counters.setdefault(type(node), 0)
                index += 1
                counters[type(node)] = index
                myself = "%s %d" % (type(node).__name__, index)
                nodes[node] = myself
            if parent is not None:
                stream.write("\t\"%s\" -> \"%s\"\n" % (parent, myself))
            for child in node._children:
                queue.append((child, myself))
        stream.write("}\n")
        return stream

    def _get_log_name(self):
        return self.__class__.__name__

    @staticmethod
    def _format_pipeline(pipeline):
        return " -> ".join(type(n).__name__ for n in pipeline)

    def __call__(self, head):
        raise NotImplementedError()
