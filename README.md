## ast2vec

[![Build Status](https://travis-ci.org/src-d/ast2vec.svg)](https://travis-ci.org/src-d/ast2vec) [![codecov](https://codecov.io/github/src-d/ast2vec/coverage.svg?branch=develop)](https://codecov.io/gh/src-d/ast2vec) [![PyPI](https://img.shields.io/pypi/v/ast2vec.svg)](https://pypi.python.org/pypi/ast2vec)

Machine Learning models on top of Abstract Syntax Trees.

Currently, there are implemented:

* id2vec, source code identifier embeddings
* docfreq, source code identifier document frequencies (part of TF-IDF)
* nBOW, weighted bag of vectors, as in [src-d/wmd-relax](https://github.com/src-d/wmd-relax)

All the models are stored in [ASDF](http://asdf-standard.readthedocs.io/en/latest/) format.

## Install

```
pip3 install git+https://github.com/bblfsh/client-python
pip3 install ast2vec
```

## Usage

The project exposes two interfaces: API and command line. The command line is

```
python3 -m ast2vec --help
```
There is an example of using Python API  [here](Doc/how_to_use_ast2vec.ipynb).

It exposes several tools to generate the models and setup the environment.

API is divided into two domains: models and training. The first is about using while the second
is about creating. Models: [Id2Vec](ast2vec/id2vec.py),
[DocumentFrequencies](ast2vec/df.py), [NBOW](ast2vec/nbow.py), [Cooccurrences](ast2vec/coocc.py).
Transformers (keras/sklearn style): [Repo2nBOWTransformer](ast2vec/repo2/nbow.py#L72),
[Repo2CooccTransformer](ast2vec/repo2/coocc.py#L101),
[PreprocessTransformer](ast2vec/id_embedding.py#L22),
[SwivelTransformer](ast2vec/id_embedding.py#L218) and
[PostprocessTransformer](ast2vec/id_embedding.py#L241).

## Algorithms

#### Identifier embeddings

We build the source code identifier co-occurrence matrix for every repository.

1. Clone or read the repository from disk.
2. Classify files using [enry](https://github.com/src-d/enry).
3. Extract [UAST](https://doc.bblf.sh/uast/specification.html) from each supported file.
4. [Split and stem](ast2vec/repo2/base.py#L160) all the identifiers in each tree.
5. [Traverse UAST](ast2vec/repo2/coocc.py#L86), collapse all non-identifier paths and record all
identifiers on the same level as co-occurring. Besides, connect them with their immediate parents.
6. Write the individual co-occurrence matrices.
7. [Merge](ast2vec/id_embedding.py#L50) co-occurrence matrices from all repositories. Write the
document frequencies model.
8. Train the embeddings using [Swivel](ast2vec/swivel.py) running on Tensorflow. Interactively view
the intermediate results in Tensorboard using `--logs`.
9. Write the identifier embeddings model.
10. Publish generated models to the Google Cloud Storage.

1-6 is performed with `repo2coocc` tool / `Repo2CooccTransformer` class,
7 with `id2vec_preproc` / `id_embedding.PreprocessTransformer`, 8 with `id2vec_train` / `id_embedding.SwivelTransformer`,
9 with `id2vec_postproc` / `id_embedding.PostprocessTransformer` and 10 with `publish`.

#### Weighted Bag of Vectors

We represent every repository as a weighted bag-of-vectors, provided by we've got document
frequencies ("docfreq") and identifier embeddings ("id2vec").

1. Clone or read the repository from disk.
2. Classify files using [enry](https://github.com/src-d/enry).
3. Extract [UAST](https://doc.bblf.sh/uast/specification.html) from each supported file.
4. [Split and stem](ast2vec/repo2/base.py#L160) all the identifiers in each tree.
5. Leave only those identifiers which are present in "docfreq" and "id2vec".
6. Set the weight of each such identifier as TF-IDF.
7. Set the value of each such identifier as the corresponding embedding vector.
8. Write the nBOW model.
9. Publish it to the Google Cloud Storage.

1-8 is performed with `repo2nbow` tool / `Repo2nBOWTransformer` class and 9 with `publish`.

## Contributions
[![PEP8](https://img.shields.io/badge/code%20style-pep8-orange.svg)](https://www.python.org/dev/peps/pep-0008/)

We use [PEP8](https://www.python.org/dev/peps/pep-0008/) with line length 99 and ". All the tests
must pass:

```
python3 -m unittest discover /path/to/ast2vec
```

## License

Apache 2.0.
