from modelforge import Model


@register_model
class GraphDef(Model):
    """
    Id split model exported a a GraphDef protobuf.
    """
    NAME = "graph-def"

    def construct(self, graph_def):
        self._graph_def = graph_def
        return self

    @property
    def graph_def(self):
        """
        Returns the model as a GraphDef protobuf.
        """
        return self._graph_def

    def _generate_tree(self):
        return {"graph_def": self.graph_def}

    def _load_tree(self, tree):
        self.construct(graph_def=tree["graph_def"])
