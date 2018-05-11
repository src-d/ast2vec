from typing import List

from modelforge import register_model, Model


@register_model
class TensorFlowModel(Model):
    """
    TensorFlow Protobuf model exported in the Modelforge format with GraphDef inside.
    """
    NAME = "tensorflow-model"

    def construct(self, graphdef, session, outputs: List[str]):
        self._graphdef = graphdef
        self._session = session
        self._outputs = outputs
        return self

    @property
    def graphdef(self):
        """
        Returns a TensorFlow GraphDef.
        """
        return self._graphdef

    def session(self):
        """
        Returns a tensorflow session.
        """
        return self._session

    @property
    def outputs(self):
        """
        Returns the list of outputs of the tensorflow graph.
        """
        return self._outputs

    def serialize(self):
        """
        Exports the Protobuf tensorflow model into a GraphDef object
        Makes all the variables of the graph constant and serializes them to a byte string
        """
        from tensorflow.python.framework import graph_util

        for node in self._graphdef.node:
            node.device = ""
        constant_graph = graph_util.convert_variables_to_constants(self._session, self._graphdef,
                                                                   self._outputs)
        return constant_graph.SerializeToString()

    def deserialize(self):
        """
        Deserializes the GraphDef byte object saved out in the Modelforge format.
        """
        import tensorflow as tf
        from tensorflow.core.framework import graph_pb2

        graphdef = graph_pb2.GraphDef()
        graphdef.ParseFromString(self._graphdef)
        return graphdef

    def _generate_tree(self):
        return {"graphdef": self.serialize()}

    def _load_tree(self, tree):
        self.construct(graphdef=tree["graphdef"])
