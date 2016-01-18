
lazyproperty\_RO
================

A decorator for a lazy property that is readonly A lazy property is just
a normal property which is only calculated once!

.. code:: python

    from decorator_collection import lazyproperty_readonly

Let's create a class with a lazy property
=========================================

.. code:: python

    class Test(object):
        @lazyproperty_readonly
        def prop(self):
            print('Calculating lazy property.')
            return 100

This will print something every time the value is calculated so we can
use the text output if the value is calculated.

Let's create an instance and call the property

.. code:: python

    A = Test()
    A.prop


.. parsed-literal::

    Calculating lazy property.
    



.. parsed-literal::

    100



Ok, it calculated the property, what happens if we call it again?

.. code:: python

    A.prop




.. parsed-literal::

    100



It didn't calculate it again. So this is what makes it a lazy property.

But what happens behind the scenes?

Behind the scenes the value is only calculated if there is no entry in
the \_\ *dict\_* of the instance.

.. code:: python

    B = Test()
    B.__dict__




.. parsed-literal::

    {}



The dict is empty after creating a new instance, let's call the property
and view the dict again.

.. code:: python

    B.prop
    B.__dict__


.. parsed-literal::

    Calculating lazy property.
    



.. parsed-literal::

    {'prop': 100}



There the value is saved and will be read from the dict rather than
calculating it again. Ok we had no calculation but if there were it
would be only calculated once.

Readonly
========

means that a custom setter and deleter are simply ignored.

.. code:: python

    class Test_Setter_Deleter(object):
        @lazyproperty_readonly
        def prop(self):
            print('Calculating lazy property.')
            return 100
    
        @prop.setter
        def prop(self, value):
            self.__dict__['prop'] = value
    
        @prop.deleter
        def prop(self):
            if 'prop' in self.__dict__:
                del self.__dict__['prop']

Ok, let's see if it is readonly let's try to set it:

.. code:: python

    C = Test_Setter_Deleter()
    C.prop = 20


::


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-8-ffa2f2e9fe60> in <module>()
          1 C = Test_Setter_Deleter()
    ----> 2 C.prop = 20
    

    C:\Users\Admin\Master\Python_Scripts\General\decorator_collection.py in __set__(self, obj, val)
         30 
         31     def __set__(self, obj, val):
    ---> 32         raise AttributeError("can't set attribute")
         33 
         34     def __delete__(self, obj):
    

    AttributeError: can't set attribute


Did not work, let's check the dict

.. code:: python

    C.__dict__




.. parsed-literal::

    {}



Still empty, let's see if deleting it works:

.. code:: python

    C = Test_Setter_Deleter()
    C.prop
    del C.prop


.. parsed-literal::

    Calculating lazy property.
    

::


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-10-6e25872f013b> in <module>()
          1 C = Test_Setter_Deleter()
          2 C.prop
    ----> 3 del C.prop
    

    C:\Users\Admin\Master\Python_Scripts\General\decorator_collection.py in __delete__(self, obj)
         33 
         34     def __delete__(self, obj):
    ---> 35         raise AttributeError("can't delete attribute")
         36 
         37 
    

    AttributeError: can't delete attribute


.. code:: python

    C.__dict__




.. parsed-literal::

    {'prop': 100}



Delete does not work and the value is still saved in the dict. So it's
probably everything it promises.
