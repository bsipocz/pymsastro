# Licensed under a 3-clause BSD style license - see LICENSE.rst

import six

__all__ = ['lazyproperty_readonly', 'format_doc']


class lazyproperty_readonly(property):
    """
    Like `~astropy.utils.decorators.lazyproperty` in `astropy.utils.decorators`
    but does not setup any setter or deleter - they are not possible with this
    decorator.

    Decorating a function with `lazyproperty_readonly` will calculate the value
    only if the value was not calculated before and saved. The calculated value
    is saved in the objects ``__dict__`` with the same name as the function
    that is decorated.
    """

    def __init__(self, fget, fset=None, fdel=None, doc=None):
        super(lazyproperty_readonly, self).__init__(fget, fset, fdel, doc)
        self._key = self.fget.__name__

    def __get__(self, obj, owner=None):
        if obj is None:
            return self
        if self._key in obj.__dict__:
            return obj.__dict__[self._key]
        else:
            val = self.fget(obj)
            obj.__dict__[self._key] = val
            return val

    def __set__(self, obj, val):
        raise AttributeError("can't set attribute")

    def __delete__(self, obj):
        raise AttributeError("can't delete attribute")


def format_doc(docstring, *args, **kwargs):
    """

    Going to be in `astropy` ... eventually ...

    Replaces the docstring of the decorated object and then formats it.
    This decorator keeps the objects signature and only alters the saved
    ``__doc__`` property (so it cannot be used as class decorator in python2).
    The formatting works like :meth:`str.format` and if the decorated object
    already has a docstring this docstring can be included in the new
    documentation if you use the ``{__doc__}`` placeholder.
    It's primary use is if multiple functions have the same or only
    a slightly different *long* docstring and you want to DRY
    (https://de.wikipedia.org/wiki/Don%E2%80%99t_repeat_yourself).

    Parameters
    ----------
    docstring : `str` or object
        The docstring that will replace the docstring of the decorated
        object. If it is an object like a function or class it will
        take the docstring of this object. If it is a string it will use the
        string itself. One special case is if the string is ``'__doc__'`` then
        it will use the decorated functions docstring and formats it.
    args :
        passed to :meth:`str.format`.
    kwargs :
        passed to :meth:`str.format`. If the function has a (not empty)
        docstring the original docstring is added to the kwargs with the
        keyword ``"__doc__"``.

    Raises
    ------
    ValueError
        If the ``docstring`` (or interpreted docstring if it was ``"__doc__"``
        or not a string) is empty.
    IndexError, KeyError
        If a placeholder in the (interpreted) ``docstring`` was not filled. see
        :meth:`str.format` for more information.

    Notes
    -----
    Using this decorator allows, for example Sphinx, to parse the
    correct docstring.
    There might be problems with certain objects which have a not-writable
    ``__doc__`` attribute. Like classes in python2.

    Examples
    --------
    Replacing the current docstring is very easy::

        >>> from pymsastro.utils import format_doc
        >>> doc = '''Perform num1 + num2'''
        >>> @format_doc(doc)
        ... def add(num1, num2):
        ...     return num1+num2
        ...
        >>> help(add) # doctest: +SKIP
        Help on function add in module __main__:
        <BLANKLINE>
        add(num1, num2)
            Perform num1 + num2

    sometimes instead of replacing you only want to add to it::

        >>> doc = '''
        ...       {__doc__}
        ...       Parameters
        ...       ----------
        ...       num1, num2 : Numbers
        ...       Returns
        ...       -------
        ...       result: Number
        ...       '''
        >>> @format_doc(doc)
        ... def add(num1, num2):
        ...     '''Perform addition.'''
        ...     return num1+num2
        ...
        >>> help(add) # doctest: +SKIP
        Help on function add in module __main__:
        <BLANKLINE>
        add(num1, num2)
            Perform addition.
            Parameters
            ----------
            num1, num2 : Numbers
            Returns
            -------
            result: Number

    in case one might want to format it further::

        >>> doc = '''
        ...       Perform {0}.
        ...       Parameters
        ...       ----------
        ...       num1, num2 : Numbers
        ...       Returns
        ...       -------
        ...       result: Number
        ...           result of num1 {op} num2
        ...       {__doc__}
        ...       '''
        >>> @format_doc(doc, 'addition', op='+')
        ... def add(num1, num2):
        ...     return num1+num2
        ...
        >>> @format_doc(doc, 'subtraction', op='-')
        ... def subtract(num1, num2):
        ...     '''Notes: This one has additional notes.'''
        ...     return num1-num2
        ...
        >>> help(add) # doctest: +SKIP
        Help on function add in module __main__:
        <BLANKLINE>
        add(num1, num2)
            Perform addition.
            Parameters
            ----------
            num1, num2 : Numbers
            Returns
            -------
            result: Number
                result of num1 + num2
        >>> help(subtract) # doctest: +SKIP
        Help on function subtract in module __main__:
        <BLANKLINE>
        subtract(num1, num2)
            Perform subtraction.
            Parameters
            ----------
            num1, num2 : Numbers
            Returns
            -------
            result: Number
                result of num1 - num2
            Notes: This one has additional notes.

    These methods can be combined an even taking the docstring from another
    object is possible as docstring attribute. You just have to specify the
    object::

        >>> @format_doc(add)
        ... def another_add(num1, num2):
        ...     return num1 + num2
        ...
        >>> help(another_add) # doctest: +SKIP
        Help on function another_add in module __main__:
        <BLANKLINE>
        another_add(num1, num2)
            Perform addition.
            Parameters
            ----------
            num1, num2 : Numbers
            Returns
            -------
            result: Number
                result of num1 + num2

    But be aware that this decorator *only* formats the given docstring not
    the strings passed as ``args`` or ``kwargs`` (not even the original
    docstring)::

        >>> @format_doc(doc, 'addition', op='+')
        ... def yet_another_add(num1, num2):
        ...    '''This one is good for {0}.'''
        ...    return num1 + num2
        ...
        >>> help(yet_another_add) # doctest: +SKIP
        Help on function yet_another_add in module __main__:
        <BLANKLINE>
        yet_another_add(num1, num2)
            Perform addition.
            Parameters
            ----------
            num1, num2 : Numbers
            Returns
            -------
            result: Number
                result of num1 + num2
            This one is good for {0}.

    To work around it you could specify the docstring to be ``'__doc__'``::

        >>> @format_doc('__doc__', 'addition')
        ... def last_add_i_swear(num1, num2):
        ...    '''This one is good for {0}.'''
        ...    return num1 + num2
        ...
        >>> help(last_add_i_swear) # doctest: +SKIP
        Help on function last_add_i_swear in module __main__:
        <BLANKLINE>
        last_add_i_swear(num1, num2)
            This one is good for addition.

    Using it with ``'__doc__'`` as docstring allows to use the decorator twice
    on an object to first parse the new docstring and then to parse the
    original docstring or the ``args`` and ``kwargs``.
    """
    def set_docstring(func):
        # Not a string so assume we want the saved doc from the object
        if not isinstance(docstring, six.string_types):
            doc = docstring.__doc__
        elif docstring != '__doc__':
            doc = docstring
        else:
            doc = func.__doc__
            # Delete documentation in this case so we don't end up with
            # awkwardly self-inserted docs.
            func.__doc__ = None

        if not doc:
            # In case the docstring is empty it's probably not what was wanted.
            raise ValueError('docstring must be a string or containing a '
                             'docstring that is not empty.')

        # If the original has a not-empty docstring append it to the format
        # kwargs.
        kwargs['__doc__'] = func.__doc__ or ''
        func.__doc__ = doc.format(*args, **kwargs)
        return func
    return set_docstring
