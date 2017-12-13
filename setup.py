from importlib.machinery import SourceFileLoader
from setuptools import setup, find_packages
import sys

sourcedml = SourceFileLoader("sourced.ml", "./sourced/ml/__init__.py").load_module()

if sys.version_info < (3, 5, 0):
    typing = ["typing"]
else:
    typing = []

setup(
    name="sourced-ml",
    description="Part of source{d}'s stack for machine learning on source "
                "code. Provides API and tools to train and use models based "
                "on source code identifiers extracted from Babelfish's UASTs.",
    version=".".join(map(str, sourcedml.__version__)),
    license="Apache 2.0",
    author="source{d}",
    author_email="machine-learning@sourced.tech",
    url="https://github.com/src-d/ml",
    download_url="https://github.com/src-d/ml",
    packages=find_packages(exclude=("sourced.ml.tests",)),
    namespace_packages=['sourced'],
    entry_points={
        "console_scripts": ["sourcedml=sourced.ml.__main__:main"],
    },
    keywords=["machine learning on source code", "word2vec", "id2vec",
              "github", "swivel", "nbow", "bblfsh", "babelfish"],
    install_requires=["PyStemmer>=1.3,<2.0",
                      "bblfsh>-2.2.1,<3.0",
                      "modelforge>=0.5.0-alpha",
                      "sourced-engine>=0.2.0,<0.3",
                      "parquet>=1.2,<2.0"] + typing,
    extras_require={
        "tf": ["tensorflow>=1.0,<2.0"],
        "tf_gpu": ["tensorflow-gpu>=1.0,<2.0"],
    },
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
