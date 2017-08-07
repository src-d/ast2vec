import sys

from setuptools import setup, find_packages

if sys.version_info < (3, 5, 0):
    typing = ["typing"]
else:
    typing = []

setup(
    name="ast2vec",
    description="Part of source{d}'s stack for machine learning on source "
                "code. Provides API and tools to train and use models based "
                "on source code identifiers extracted from Babelfish's UASTs.",
    version="0.1.2-alpha",
    license="Apache 2.0",
    author="source{d}",
    author_email="machine-learning@sourced.tech",
    url="https://github.com/src-d/ast2vec",
    download_url="https://github.com/src-d/ast2vec",
    packages=find_packages(exclude=("ast2vec.tests",)),
    keywords=["machine learning on source code", "word2vec", "id2vec",
              "github", "swivel", "nbow", "bblfsh", "babelfish"],
    install_requires=["PyStemmer>=1.3,<2.0",
                      "numpy>=1.12,<2.0",
                      "scipy>=0.17,<1.0",
                      "tensorflow>=1.0,<2.0",
                      "clint>=0.5.0",
                      "asdf>=1.2,<2.0",
                      "google-cloud-storage>=1.0,<2.0",
                      "python-dateutil",
                      "modelforge>=0.2.3-alpha",
                      "bblfsh>=0.0.4"] + typing,
    package_data={"": ["LICENSE", "README.md"]},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Topic :: Software Development :: Libraries"
    ]
)
