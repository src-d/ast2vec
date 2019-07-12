from importlib.machinery import SourceFileLoader
import io
import os.path

from setuptools import find_packages, setup

sourcedml = SourceFileLoader("sourced.ml", "./sourced/ml/__init__.py").load_module()

with io.open(os.path.join(os.path.dirname(__file__), "README.md"),
             encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="sourced-ml",
    description="Framework for machine learning on source code. "
                "Provides API and tools to train and use models based "
                "on source code features extracted from Babelfish's UASTs.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=sourcedml.__version__,
    license="Apache 2.0",
    author="source{d}",
    author_email="machine-learning@sourced.tech",
    url="https://github.com/src-d/ml",
    download_url="https://github.com/src-d/ml",
    packages=find_packages(exclude=("sourced.ml.tests",)),
    namespace_packages=["sourced"],
    entry_points={
        "console_scripts": ["srcml=sourced.ml.__main__:main"],
    },
    keywords=["machine learning on source code", "word2vec", "id2vec",
              "github", "swivel", "bow", "bblfsh", "babelfish"],
    install_requires=[
        "PyStemmer>=1.3,<2.0",
        "bblfsh>=2.2.1,<3.0",
        "modelforge>=0.11.1,<0.14",
        "sourced-jgit-spark-connector>=2.0.1,<2.1.0",
        "humanize>=0.5.0,<0.6",
        "parquet>=1.2,<2.0",
        "pygments>=2.2.0,<3.0",
        "keras>=2.0,<3.0",
        "scikit-learn>=0.19,<1.0",
        "tqdm>=4.20,<5.0",
        "typing;python_version<'3.5'",
    ],
    extras_require={
        "tf": ["tensorflow>=1.0,<2.0"],
        "tf_gpu": ["tensorflow-gpu>=1.0,<2.0"],
        "pandas": ["pandas>=0.24.1,<1.0"],
    },
    tests_require=["docker>=3.6.0,<4.0"],
    package_data={"": ["LICENSE.md", "README.md"],
                  "sourced": ["ml/transformers/languages.yml"], },
    python_requires=">=3.4",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Software Development :: Libraries"
    ]
)
