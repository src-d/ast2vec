import argparse
import sys
import unittest


import ast2vec.__main__ as main
import ast2vec.model2.join_bow


class MainTests(unittest.TestCase):
    def test_handlers(self):
        handlers = [False] * 11

        def repo2nbow_entry(args):
            handlers[0] = True

        def repos2nbow_entry(args):
            handlers[1] = True

        def repo2coocc_entry(args):
            handlers[2] = True

        def repos2coocc_entry(args):
            handlers[3] = True

        def joinbow_entry(args):
            handlers[4] = True

        def preprocess(args):
            handlers[5] = True

        def run_swivel(args):
            handlers[6] = True

        def postprocess(args):
            handlers[7] = True

        def bow2vw_entry(args):
            handlers[8] = True

        def install_enry(args):
            handlers[9] = True

        def dump_model(args):
            handlers[10] = True

        main.repo2nbow_entry = repo2nbow_entry
        main.repos2nbow_entry = repos2nbow_entry
        main.repo2coocc_entry = repo2coocc_entry
        main.repos2coocc_entry = repos2coocc_entry
        main.joinbow_entry = joinbow_entry
        main.preprocess = preprocess
        main.run_swivel = run_swivel
        main.postprocess = postprocess
        main.bow2vw_entry = bow2vw_entry
        main.install_enry = install_enry
        main.dump_model = dump_model
        args = sys.argv
        error = argparse.ArgumentParser.error
        argparse.ArgumentParser.error = lambda self, message: None

        for action in ("repo2nbow", "repos2nbow", "repo2coocc", "repos2coocc", "join_bow",
                       "id2vec_preproc", "id2vec_train", "id2vec_postproc", "bow2vw",
                       "enry", "dump"):
            sys.argv = [main.__file__, action]
            main.main()

        sys.argv = args
        argparse.ArgumentParser.error = error
        self.assertEqual(sum(handlers), len(handlers))

if __name__ == "__main__":
    unittest.main()
