import os


class DanglingSymlinkError(Exception):
    """
    Exception for :func:`resolve_symlink` when a symlink points to a nonexistent file.
    """
    pass


def resolve_symlink(path: str) -> str:
    """
    Resolve the symlink if path is a symbolic link.
    Check file existence.

    :param path: path to check
    :return: filepath to existing file
    :raise DanglingSymlinkError: The symlink is broken.
    """
    # Check if file path is path
    # sometimes you have path to nonexistent files. We need to check it
    islink = os.path.islink(path)
    filepath = os.readlink(path) if islink else path
    if not os.path.exists(filepath):
        err_msg = "File %s does not exist.\n\t" % filepath + \
                  "Get it from %s path." % path if islink else "And it is not a path..."
        raise DanglingSymlinkError(err_msg)
    return filepath
