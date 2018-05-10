from modelforge import register_model, Model


@register_model
class TensorModel(Model):
    """
    """
    NAME = "tensor-model"

    def construct(self, tensor_model):
        self._tensor_model = tensor_model
        return self

    @property
    def tensor_model(self):
        """
        """
        return self._tensor_model

    def serialize(self):
        from keras import backend
        import tensorflow as tf
        from tensorflow.python.framework import graph_util, graph_io
        
        session = backend.get_session()
        tf.identity(self._tensor_model.outputs[0], name="output")
        graph_def = session.graph.as_graph_def()
        for node in graph_def.node:
            node.device = ""
        constant_graph = graph_util.convert_variables_to_constants(session, graph_def, ["output"])
        return constant_graph.SerializeToString()

    def deserialize(self):
        """
        TODO
        """

    def _generate_tree(self):
        return {"tensor_model": serialize(self._tensor_model)}

    def _load_tree(self, tree):
        self.construct(tensor_model=tree["tensor_model"])
