# Licensed under a 3-clause BSD style license - see LICENSE.rst

try:
    from ujson import dump, load
except ImportError:
    from json import dump, load

__all__ = ['json_write', 'json_append', 'json_read']


def json_write(filename, content):
    """
    Just a convenience function for `json` dump by writing/replacing a file.

    Parameters
    ----------
    filename : `str`
        The filename in which to dump the content.

    content : `list`-like, `dict`-like, ... ?
        The content that should be saved. Must be dumpable by the json module.
    """
    with open(filename, 'w') as file:
        dump(content, file)


def json_append(filename, content):
    """
    Just a convenience function for `json` dump by appending to a file.

    Parameters
    ----------
    filename : `str`
        The filename in which to dump the content.

    content : `list`-like, `dict`-like, ... ?
        The content that should be saved. Must be dumpable by the json module.
    """
    with open(filename, 'a') as file:
        dump(content, file)


def json_read(filename):
    """
    Just a convenience function for `json` load.

    Parameters
    ----------
    filename : `str`
        The filename from which to read the content.

    Returns
    -------
    content : `list`-like, `dict`-like, ... ?
        The content that was saved.
    """
    with open(filename, 'r') as file:
        res = load(file)
    return res
