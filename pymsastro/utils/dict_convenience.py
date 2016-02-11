# Licensed under a 3-clause BSD style license - see LICENSE.rst

__all__ = ['dictKeysInDict', 'dictKeysInDictFirst', 'dictKeysInDictValueFirst']


def dictKeysInDict(dictionary, keys):
    """
    Checks if any of a list of keys is in the dict and returns which are.

    Parameters
    ----------
    keys : ``iterable``
        The keys to check.

    dictionary : `dict`-like
        The dict in which to look for the keys

    Returns
    -------
    found_keys : `list`
        The keys of ``keys`` that were found in the ``dictionary``

    Raises
    ------
    KeyError
        If no key of the list of ``keys`` is in the ``dictionary``
    """
    found = []
    for i in keys:
        if i in dictionary:
            found.append(i)

    if not found:
        raise KeyError('No key of {0} was found in the dict.'.format(keys))

    return found


def dictKeysInDictFirst(dictionary, keys):
    """
    Identical to :func:`dictKeysInDict` but only returns the first found
    key.

    Returns
    -------
    first_found_key : same type as element of ``keys``
        The first key that was found in the ``dictionary``.

    Notes
    -----
    This function iterates over ``keys`` so the first key in ``keys`` that is
    a key in ``dictionary`` is the first found key. The order of the
    ``dictionary`` is ignored (because it is mostly unordered).
    """
    return dictKeysInDict(dictionary, keys)[0]


def dictKeysInDictValueFirst(dictionary, keys):
    """
    Identical to :func:`dictKeysInDict` but returns the
    associated value of the first found key.

    Returns
    -------
    first_found_key_value : same type as element of ``dictionary``
        The value for the first key that was found in the ``dictionary``.

    Notes
    -----
    This function iterates over ``keys`` so the first key in ``keys`` that is
    a key in ``dictionary`` is the first found key. The order of the
    ``dictionary`` is ignored (because it is mostly unordered).
    """
    return dictionary[dictKeysInDictFirst(dictionary, keys)]
