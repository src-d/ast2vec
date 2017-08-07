import argparse
import logging

from modelforge.progress_bar import progress_bar

from ast2vec.id2vec import Id2Vec
from ast2vec.bow import BOW, NBOW


def convert_bow_to_vw(bow: BOW, output: str):
    log = logging.getLogger("bow2vw")
    log.info("Writing %s", output)
    with open(output, "w") as fout:
        for index in progress_bar(bow, log, expected_size=len(bow)):
            record = bow[index]
            fout.write(record[0] + " ")
            pairs = []
            for t, v in zip(*record[1:]):
                try:
                    word = bow.tokens[t]
                except (KeyError, IndexError):
                    log.warning("%d not found in the vocabulary", t)
                    continue
                pairs.append("%s:%s" % (word, v))
            fout.write(" ".join(pairs))
            fout.write("\n")


def bow2vw_entry(args: argparse.Namespace):
    if not args.nbow:
        bow = BOW().load(source=args.bow)
    else:
        bow = NBOW.as_bow(args.nbow, args.id2vec)
    convert_bow_to_vw(bow, args.output)
