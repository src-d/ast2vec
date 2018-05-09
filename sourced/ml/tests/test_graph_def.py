import unittest

from tensorflow.core.framework import graph_pb2

import sourced.ml.tests.models as paths
from sourced.ml.models import GraphDef


class GraphDefTests(unittest.TestCase):
    def setUp(self):
        self.model = GraphDef().load(source=paths.GRAPHDEF)
        self.graph_def = graph_pb2.GraphDef()
        self.graph_def.ParseFromString(self.model.graph_def)

    def test_graph_def(self):
        self.assertTrue(self.graph_def.ByteSize() > 0)
        self.assertTrue(self.graph_def.IsInitialized())
        self.assertIsInstance(self.graph_def.SerializeToString(), bytes)
        self.assertEqual(len(self.graph_def.node), 8)

    def test_names(self):
        self.assertEqual(self.graph_def.node[0].name, "const")
        self.assertEqual(self.graph_def.node[1].name, "b")

    def test_values(self):
        self.assertEqual(self.graph_def.node[0].attr["value"].tensor.int_val[0], 2)
        self.assertEqual(self.graph_def.node[1].attr["value"].tensor.int_val[0], 5)


if __name__ == "__main__":
    unittest.main()
