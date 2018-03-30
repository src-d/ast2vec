import argparse
import sys
import unittest

import sourced.ml.__main__ as main
from sourced.ml.__main__ import ArgumentDefaultsHelpFormatterNoNone

from sourced.ml.tests.test_dump import captured_output


class MainTests(unittest.TestCase):
    def test_handlers(self):
        action2handler = {
            "id2vec_preproc": "preprocess_id2vec",
            "id2vec_train": "run_swivel",
            "id2vec_postproc": "postprocess_id2vec",
            "id2vec_project": "projector_entry",
            "bigartm2asdf": "bigartm2asdf_entry",
            "bow2vw": "bow2vw_entry",
            "bigartm": "install_bigartm",
            "dump": "dump_model",
            "repos2coocc": "repos2coocc_entry",
            "repos2df": "repos2df_entry",
            "repos2ids": "repos2ids_entry",
            "repos2bow": "repos2bow_entry",
            "repos2roles_ids": "repos2roles_and_ids_entry",
            "repos2id_distance": "repos2id_distance_entry"
        }
        parser = main.get_parser()
        subcommands = set([x.dest for x in parser._subparsers._actions[2]._choices_actions])
        set_action2handler = set(action2handler)
        self.assertFalse(len(subcommands - set_action2handler),
                         "You forgot to add to this test {} subcommand(s) check".format(
                             subcommands - set_action2handler))

        self.assertFalse(len(set_action2handler - subcommands),
                         "You cover unexpected subcommand(s) {}".format(
                             set_action2handler - subcommands))

        called_actions = []
        args_save = sys.argv
        error_save = argparse.ArgumentParser.error
        try:
            argparse.ArgumentParser.error = lambda self, message: None

            for action, handler in action2handler.items():
                def handler_append(*args, **kwargs):
                    called_actions.append(action)

                handler_save = getattr(main, handler)
                try:
                    setattr(main, handler, handler_append)
                    sys.argv = [main.__file__, action]
                    main.main()
                finally:
                    setattr(main, handler, handler_save)
        finally:
            sys.argv = args_save
            argparse.ArgumentParser.error = error_save

        set_called_actions = set(called_actions)
        set_actions = set(action2handler)
        self.assertEqual(set_called_actions, set_actions)
        self.assertEqual(len(set_called_actions), len(called_actions))

    def test_empty(self):
        args = sys.argv
        error = argparse.ArgumentParser.error
        try:
            argparse.ArgumentParser.error = lambda self, message: None

            sys.argv = [main.__file__]
            with captured_output() as (stdout, _, _):
                main.main()
        finally:
            sys.argv = args
            argparse.ArgumentParser.error = error
        self.assertIn("usage:", stdout.getvalue())

    def test_custom_formatter(self):
        class FakeAction:
            default = None
            option_strings = ['--param']
            nargs = None
            help = "help"

        formatter = ArgumentDefaultsHelpFormatterNoNone(None)
        help = formatter._expand_help(FakeAction)
        self.assertEqual("help", help)
        FakeAction.default = 10
        help = formatter._expand_help(FakeAction)
        self.assertEqual("help (default: 10)", help)


if __name__ == "__main__":
    unittest.main()
