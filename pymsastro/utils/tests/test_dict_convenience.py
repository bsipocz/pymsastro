# Licensed under a 3-clause BSD style license - see LICENSE.rst

from ..dict_convenience import *

from pytest import raises


def test_dictKeysInDict_failure():
    dictionary = {'a': 10, 'b': 100, 'c': 1000}
    keys = ['d']
    with raises(KeyError):
        dictKeysInDict(keys, dictionary)


def test_dictKeysInDict():
    dictionary = {'a': 10, 'b': 100, 'c': 1000}

    keys = ['a']
    found = dictKeysInDict(keys, dictionary)
    assert len(found) == 1
    assert found[0] == 'a'

    keys = ['b']
    found = dictKeysInDict(keys, dictionary)
    assert len(found) == 1
    assert found[0] == 'b'

    keys = ['c']
    found = dictKeysInDict(keys, dictionary)
    assert len(found) == 1
    assert found[0] == 'c'

    keys = ['a', 'b']
    found = dictKeysInDict(keys, dictionary)
    assert len(found) == 2
    assert found[0] == 'a'
    assert found[1] == 'b'

    keys = ['a', 'c']
    found = dictKeysInDict(keys, dictionary)
    assert len(found) == 2
    assert found[0] == 'a'
    assert found[1] == 'c'

    keys = ['b', 'c']
    found = dictKeysInDict(keys, dictionary)
    assert len(found) == 2
    assert found[0] == 'b'
    assert found[1] == 'c'

    keys = ['a', 'b', 'c']
    found = dictKeysInDict(keys, dictionary)
    assert len(found) == 3
    assert found[0] == 'a'
    assert found[1] == 'b'
    assert found[2] == 'c'


def test_dictKeysInDictFirst():
    dictionary = {'a': 10, 'b': 100, 'c': 1000}

    keys = ['d']
    with raises(KeyError):
        dictKeysInDictFirst(keys, dictionary)

    keys = ['a']
    found = dictKeysInDictFirst(keys, dictionary)
    assert found == 'a'

    keys = ['b']
    found = dictKeysInDictFirst(keys, dictionary)
    assert found == 'b'

    keys = ['c']
    found = dictKeysInDictFirst(keys, dictionary)
    assert found == 'c'

    keys = ['a', 'b']
    found = dictKeysInDictFirst(keys, dictionary)
    assert found == 'a'

    keys = ['a', 'c']
    found = dictKeysInDictFirst(keys, dictionary)
    assert found == 'a'

    keys = ['b', 'c']
    found = dictKeysInDictFirst(keys, dictionary)
    assert found == 'b'

    keys = ['a', 'b', 'c']
    found = dictKeysInDictFirst(keys, dictionary)
    assert found == 'a'


def test_dictKeysInDictValueFirst():
    dictionary = {'a': 10, 'b': 100, 'c': 1000}

    keys = ['d']
    with raises(KeyError):
        dictKeysInDictValueFirst(keys, dictionary)

    keys = ['a']
    found = dictKeysInDictValueFirst(keys, dictionary)
    assert found == 10

    keys = ['b']
    found = dictKeysInDictValueFirst(keys, dictionary)
    assert found == 100

    keys = ['c']
    found = dictKeysInDictValueFirst(keys, dictionary)
    assert found == 1000

    keys = ['a', 'b']
    found = dictKeysInDictValueFirst(keys, dictionary)
    assert found == 10

    keys = ['a', 'c']
    found = dictKeysInDictValueFirst(keys, dictionary)
    assert found == 10

    keys = ['b', 'c']
    found = dictKeysInDictValueFirst(keys, dictionary)
    assert found == 100

    keys = ['a', 'b', 'c']
    found = dictKeysInDictValueFirst(keys, dictionary)
    assert found == 10
