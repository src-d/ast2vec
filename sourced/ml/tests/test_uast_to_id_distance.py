import unittest

from bblfsh import BblfshClient

from sourced.ml.algorithms import Uast2IdTreeDistance, Uast2IdLineDistance, NoopTokenParser
from sourced.ml.tests.models import SOURCE_PY


class Uast2IdTreeDistanceTest(unittest.TestCase):
    def setUp(self):
        self.uast2role_id_pairs = Uast2IdTreeDistance(token_parser=NoopTokenParser(),
                                                      max_distance=4)
        self.uast = BblfshClient("0.0.0.0:9432").parse(SOURCE_PY).uast

    def test_result(self):
        correct = [(('ModuleSpec', '__spec__'), 2),
                   (('ModuleSpec', 'namedtuple'), 3),
                   (('ModuleSpec', 'utmain'), 3),
                   (('__package__', '__spec__'), 2),
                   (('__package__', 'utmain'), 3),
                   (('__package__', 'utmain'), 3),
                   (('__spec__', 'ModuleSpec'), 3),
                   (('__spec__', 'ModuleSpec'), 2),
                   (('__spec__', 'utmain'), 3),
                   (('__spec__', 'utmain'), 3),
                   (('collections', 'ModuleSpec'), 2),
                   (('collections', 'ModuleSpec'), 3),
                   (('collections', 'ModuleSpec'), 2),
                   (('collections', '__spec__'), 2),
                   (('collections', 'namedtuple'), 2),
                   (('collections', 'namedtuple'), 3),
                   (('collections', 'utmain'), 3),
                   (('modelforge.logs', 'modules'), 3),
                   (('modelforge.logs', 'setup'), 3),
                   (('modelforge.logs', 'setup_logging'), 2),
                   (('modelforge.logs', 'utmain'), 2),
                   (('modelforge.logs', 'utmain'), 2),
                   (('modules', 'sys'), 3),
                   (('modules', 'utmain'), 3),
                   (('namedtuple', 'ModuleSpec'), 2),
                   (('namedtuple', 'ModuleSpec'), 3),
                   (('namedtuple', 'ModuleSpec'), 2),
                   (('namedtuple', 'ModuleSpec'), 3),
                   (('namedtuple', '__spec__'), 2),
                   (('namedtuple', '__spec__'), 3),
                   (('namedtuple', 'utmain'), 3),
                   (('setup_logging', 'modules'), 3),
                   (('setup_logging', 'setup'), 3),
                   (('setup_logging', 'utmain'), 2),
                   (('setup_logging', 'utmain'), 2),
                   (('sys', 'modelforge.logs'), 2),
                   (('sys', 'modules'), 3),
                   (('sys', 'setup'), 3),
                   (('sys', 'setup_logging'), 2),
                   (('sys', 'utmain'), 2),
                   (('sys', 'utmain'), 2),
                   (('utmain', 'ModuleSpec'), 2),
                   (('utmain', 'ModuleSpec'), 3),
                   (('utmain', '__spec__'), 3),
                   (('utmain', 'modules'), 3),
                   (('utmain', 'setup'), 3),
                   (('utmain', 'setup'), 3)]

        res = sorted(self.uast2role_id_pairs(self.uast), key=lambda x: x[0])
        self.assertEqual(res, correct)


class Uast2IdLineDistanceTest(unittest.TestCase):
    def setUp(self):
        self.uast2role_id_pairs = Uast2IdLineDistance(token_parser=NoopTokenParser(),
                                                      max_distance=3)
        self.uast = BblfshClient("0.0.0.0:9432").parse(SOURCE_PY).uast

    def test_result(self):
        correct = [(('ModuleSpec', '__package__'), 2),
                   (('ModuleSpec', '__spec__'), 1),
                   (('ModuleSpec', '__spec__'), 2),
                   (('ModuleSpec', 'namedtuple'), 0),
                   (('ModuleSpec', 'utmain'), 1),
                   (('ModuleSpec', 'utmain'), 2),
                   (('ModuleSpec', 'utmain'), 2),
                   (('ModuleSpec', 'utmain'), 2),
                   (('ModuleSpec', 'utmain'), 1),
                   (('__package__', '__spec__'), 0),
                   (('__package__', 'utmain'), 0),
                   (('__package__', 'utmain'), 0),
                   (('__spec__', 'ModuleSpec'), 0),
                   (('__spec__', 'ModuleSpec'), 1),
                   (('__spec__', 'utmain'), 0),
                   (('__spec__', 'utmain'), 2),
                   (('__spec__', 'utmain'), 0),
                   (('collections', 'ModuleSpec'), 1),
                   (('collections', 'ModuleSpec'), 2),
                   (('collections', '__package__'), 1),
                   (('collections', '__spec__'), 2),
                   (('collections', '__spec__'), 1),
                   (('collections', 'namedtuple'), 0),
                   (('collections', 'namedtuple'), 1),
                   (('collections', 'utmain'), 2),
                   (('collections', 'utmain'), 1),
                   (('collections', 'utmain'), 1),
                   (('modelforge.logs', 'setup_logging'), 0),
                   (('modules', '__package__'), 1),
                   (('modules', '__spec__'), 1),
                   (('modules', 'collections'), 2),
                   (('modules', 'namedtuple'), 2),
                   (('modules', 'sys'), 0),
                   (('modules', 'utmain'), 1),
                   (('modules', 'utmain'), 1),
                   (('namedtuple', 'ModuleSpec'), 1),
                   (('namedtuple', 'ModuleSpec'), 2),
                   (('namedtuple', 'ModuleSpec'), 1),
                   (('namedtuple', 'ModuleSpec'), 2),
                   (('namedtuple', '__package__'), 1),
                   (('namedtuple', '__package__'), 2),
                   (('namedtuple', '__spec__'), 2),
                   (('namedtuple', '__spec__'), 1),
                   (('namedtuple', '__spec__'), 1),
                   (('namedtuple', '__spec__'), 2),
                   (('namedtuple', 'utmain'), 2),
                   (('namedtuple', 'utmain'), 1),
                   (('namedtuple', 'utmain'), 1),
                   (('namedtuple', 'utmain'), 1),
                   (('namedtuple', 'utmain'), 2),
                   (('namedtuple', 'utmain'), 2),
                   (('setup', 'setup_logging'), 1),
                   (('sys', '__package__'), 1),
                   (('sys', '__spec__'), 1),
                   (('sys', 'collections'), 2),
                   (('sys', 'modelforge.logs'), 2),
                   (('sys', 'namedtuple'), 2),
                   (('sys', 'setup_logging'), 2),
                   (('sys', 'utmain'), 1),
                   (('sys', 'utmain'), 1),
                   (('utmain', 'ModuleSpec'), 0),
                   (('utmain', 'ModuleSpec'), 1),
                   (('utmain', '__package__'), 1),
                   (('utmain', '__spec__'), 1),
                   (('utmain', '__spec__'), 0),
                   (('utmain', 'collections'), 2),
                   (('utmain', 'modules'), 0),
                   (('utmain', 'namedtuple'), 2),
                   (('utmain', 'sys'), 0)]

        res = sorted(self.uast2role_id_pairs(self.uast), key=lambda x: x[0])
        self.assertEqual(res, correct)


if __name__ == '__main__':
    unittest.main()
