import glob
import logging
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile

import numpy
from scipy.sparse import csr_matrix

from sourced.ml.models import Topics


def execute(cmd, cwd, log):
    log.info(">>> %s", cmd)
    parsed = [v for v in cmd.split(" ") if v]
    subprocess.check_call(parsed, cwd=cwd)


def install_bigartm(args=None, target="./bigartm", tempdir=None, warn_exists=True):
    """
    Deploys bigartm/bigartm at the specified path.

    :param args: :class:`argparse.Namespace` with "output" and "tmpdir". \
                 "output" sets the target directory, "tmpdir" sets \
                 the temporary directory which is used to clone bigartm/bigartm \
                 and build it.
    :param target: The path to the built executable. If args is not None, it \
                   becomes overridden.
    :param tempdir: The temporary directory where to clone and build \
                    bigartm/bigartm. If args is not None, it becomes overridden.
    :return: None if successful; otherwise, the error code (can be 0!).
    """
    log = logging.getLogger("bigartm")
    if args is not None:
        tempdir = args.tmpdir
        target = os.path.join(args.output, "bigartm")
    if shutil.which(os.path.basename(target)) or shutil.which(target, path=os.getcwd()):
        if warn_exists:
            log.warning("bigartm is in the PATH, no-op.")
        return 0
    if not shutil.which("cmake"):
        log.error("You need to install cmake.")
        return 1
    parent_dir = os.path.dirname(target)
    os.makedirs(parent_dir, exist_ok=True)
    if not os.path.isdir(parent_dir):
        log.error("%s is not a directory.", parent_dir)
        return 2
    with tempfile.TemporaryDirectory(prefix="bigartm-", dir=tempdir) as tmpdir:
        log.info("Building bigartm/bigartm in %s...", tmpdir)
        execute("git clone --single-branch --depth=1 https://github.com/bigartm/bigartm .",
                tmpdir, log)
        cwd = os.path.join(tmpdir, "build")
        os.mkdir(cwd)
        execute("cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DPYTHON=python3 ..",
                cwd, log)
        execute("make -j%d" % multiprocessing.cpu_count(), cwd, log)
        whl_path = glob.glob(os.path.join(tmpdir, "build/python/*.whl"))[0]
        execute("pip3 install \"%s\"" % whl_path, cwd, log)
        shutil.copyfile(os.path.join(cwd, "bin", "bigartm"), target)
        os.chmod(target, 0o777)
    log.info("Installed %s", os.path.abspath(target))


def bigartm2asdf_entry(args):
    """
    BigARTM "readable" model -> Topics -> Modelforge ASDF.
    """
    tokens = []
    data = []
    indices = []
    indptr = [0]
    if args.input != "-":
        fin = open(args.input)
    else:
        fin = sys.stdin
    try:
        # the first line is the header
        fin.readline()
        for line in fin:
            items = line.split(";")
            tokens.append(items[0])
            nnz = 0
            for i, v in enumerate(items[2:]):
                if v == "0":
                    continue
                nnz += 1
                data.append(float(v))
                indices.append(i)
            indptr.append(indptr[-1] + nnz)
    finally:
        if args.input != "-":
            fin.close()
    data = numpy.array(data, dtype=numpy.float32)
    indices = numpy.array(indices, dtype=numpy.int32)
    matrix = csr_matrix((data, indices, indptr), shape=(len(tokens), len(items) - 2)).T
    Topics().construct(tokens, None, matrix).save(args.output)
