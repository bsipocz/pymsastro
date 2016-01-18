
Signal to Noise Ratio Measurements
==================================

Very often we need the signal to noise ratio as indicator how good the
spectrum quality is. There are several methods to calculate it.

.. code:: python

    from signal_to_noise_calculator import *
    
    import numpy as np

Let's create a signal of mean 100 with normal distributed noise with
standard deviation 2. This should result in a theoretical
signal-to-noise ratio of 100/2 = 50.

.. code:: python

    signal_with_noise = np.random.normal(100,2,100000)

DER\_SNR
--------

One way of measuring the signal to noise based on median statistics is
the DER\_SNR scripy from
http://www.stecf.org/software/ASTROsoft/DER\_SNR/ . Let us see what it
determines:

.. code:: python

    snr1 = DER_SNR(signal_with_noise)
    print('signal: {0}'.format(snr1.signal))
    print('noise : {0}'.format(snr1.noise))
    print('snr   : {0}'.format(snr1.snr))


.. parsed-literal::

    signal: 100.00934671205681
    noise : 1.9970552755883146
    snr   : 50.07840690969105
    

RMSE\_SNR
---------

If one knows the theoretical value one can also compute the SNR by the
absolute and relative RMSE. I'll call this RMSE\_SNR. Since it requires
a theoretical value for the signal the signal is fixed.

*But the RMSE\_SNR requires the reference spectrum to be known!*

.. code:: python

    theoretical_signal = 100
    snr2 = RMSE_SNR(theoretical_signal, signal_with_noise)
    print('signal: {0}'.format(snr2.signal))
    print('noise : {0}'.format(snr2.noise))
    print('snr   : {0}'.format(snr2.snr))


.. parsed-literal::

    signal: 100
    noise : 2.004307258769534
    snr   : 49.892549938371765
    

For easy usage I created a metaclass that defines an interface for SNR
calculations and always provides the attributes

-  ``signal``
-  ``noise``
-  ``signal_to_noise_ratio`` (equivalent to ``snr``)

Each of these properties is lazy and readonly. (see
docorator\_collection the lazyproperty\_RO)

"DER\_SNR" as well as "RMSE\_SNR" allow additional keywords for:

-  ``ignore_zeros``: Values of exactly zero will be deleted from the
   measured array
-  ``ignore_nan_inf``: Values of NaN and Inf will be deleted.
-  ``ignore_masked``: Masked values will be deleted (the measured array
   must have an attribute "mask" for the mask and "data" for the
   measured values. This is to allow ``np.ndarray``)
-  ``verbose``: Print which values are ignored.

by default all of them are ``True``, except verbose which is ``False``

Useage of optional parameters
=============================

Quick check that the work like expected.

Ignore\_zeros:
--------------

.. code:: python

    a = np.array([100,0,101,99,100,100])
    
    print('#'*20 + ' DER ' + '#'*20)
    print('Without zeros')
    snr3 = DER_SNR(a, ignore_zeros=True)
    print(snr3.signal)
    print(snr3.noise)
    print(snr3.snr)
    
    print('With zeros')
    snr4 = DER_SNR(a, ignore_zeros=False)
    print(snr4.signal)
    print(snr4.noise)
    print(snr4.snr)
    
    print('#'*20 + ' RMSE ' + '#'*20)
    print('Without zeros')
    snr5 = RMSE_SNR(100, a, ignore_zeros=True)
    print(snr5.signal)
    print(snr5.noise)
    print(snr5.snr)
    
    print('With zeros')
    snr6 = RMSE_SNR(100, a, ignore_zeros=False)
    print(snr6.signal)
    print(snr6.noise)
    print(snr6.snr)


.. parsed-literal::

    #################### DER ####################
    Without zeros
    100.0
    1.2105394
    82.6078027696
    With zeros
    100.0
    30.263485
    3.30431211078
    #################### RMSE ####################
    Without zeros
    100
    0.632455532034
    158.113883008
    With zeros
    100
    40.8289113252
    2.44924483055
    

Ignore\_nan\_inf:
-----------------

.. code:: python

    a = np.array([100,np.nan,101,99,100,100])
    
    print('#'*20 + ' DER ' + '#'*20)
    print('Without nan')
    snr7 = DER_SNR(a, ignore_nan_inf=True)
    print(snr7.signal)
    print(snr7.noise)
    print(snr7.snr)
    
    print('With nan')
    snr8 = DER_SNR(a, ignore_nan_inf=False)
    print(snr8.signal)
    print(snr8.noise)
    print(snr8.snr)
    
    print('#'*20 + ' RMSE ' + '#'*20)
    print('Without nan')
    snr9 = RMSE_SNR(100, a, ignore_nan_inf=True)
    print(snr9.signal)
    print(snr9.noise)
    print(snr9.snr)
    
    print('With nan')
    snr10 = RMSE_SNR(100, a, ignore_nan_inf=False)
    print(snr10.signal)
    print(snr10.noise)
    print(snr10.snr)


.. parsed-literal::

    #################### DER ####################
    Without nan
    100.0
    1.2105394
    82.6078027696
    With nan
    nan
    nan
    nan
    #################### RMSE ####################
    Without nan
    100
    0.632455532034
    158.113883008
    With nan
    100
    nan
    nan
    

.. parsed-literal::

    C:\Programming\Anaconda\lib\site-packages\numpy\lib\function_base.py:3142: RuntimeWarning: Invalid value encountered in median
      RuntimeWarning)
    

.. code:: python

    a = np.array([100,np.inf,101,99,100,100])
    
    print('#'*20 + ' DER ' + '#'*20)
    print('Without inf')
    snr11 = DER_SNR(a, ignore_nan_inf=True)
    print(snr11.signal)
    print(snr11.noise)
    print(snr11.snr)
    
    print('With inf')
    snr12 = DER_SNR(a, ignore_nan_inf=False)
    print(snr12.signal)
    print(snr12.noise)
    print(snr12.snr)
    
    print('#'*20 + ' RMSE ' + '#'*20)
    print('Without inf')
    snr13 = RMSE_SNR(100, a, ignore_nan_inf=True)
    print(snr13.signal)
    print(snr13.noise)
    print(snr13.snr)
    
    print('With inf')
    snr13 = RMSE_SNR(100, a, ignore_nan_inf=False)
    print(snr13.signal)
    print(snr13.noise)
    print(snr13.snr)


.. parsed-literal::

    #################### DER ####################
    Without inf
    100.0
    1.2105394
    82.6078027696
    With inf
    100.0
    inf
    0.0
    #################### RMSE ####################
    Without inf
    100
    0.632455532034
    158.113883008
    With inf
    100
    inf
    0.0
    

Ignore\_masked:
---------------

.. code:: python

    a = np.ma.array([100,200,101,99,100,100], mask=[0,1,0,0,0,0])
    
    print('#'*20 + ' DER ' + '#'*20)
    print('Without masked')
    snr14 = DER_SNR(a, ignore_masked=True)
    print(snr14.signal)
    print(snr14.noise)
    print(snr14.snr)
    
    print('With masked')
    snr15 = DER_SNR(a, ignore_masked=False)
    print(snr15.signal)
    print(snr15.noise)
    print(snr15.snr)
    
    print('#'*20 + ' RMSE ' + '#'*20)
    print('Without masked')
    snr16 = RMSE_SNR(100, a, ignore_masked=True)
    print(snr16.signal)
    print(snr16.noise)
    print(snr16.snr)
    
    print('With masked')
    snr17 = RMSE_SNR(100, a, ignore_masked=False)
    print(snr17.signal)
    print(snr17.noise)
    print(snr17.snr)


.. parsed-literal::

    #################### DER ####################
    Without masked
    100.0
    1.2105394
    82.6078027696
    With masked
    100.0
    31.4740244
    3.17722318345
    #################### RMSE ####################
    Without masked
    100
    0.632455532034
    158.113883008
    With masked
    100
    40.8289113252
    2.44924483055
    

Verbose:
--------

.. code:: python

    a = np.ma.array([100,200,101,99,100,100], mask=[0,1,0,0,0,0])
    
    print('#'*20 + ' DER ' + '#'*20)
    snr18 = DER_SNR(a, verbose=True)
    print(snr18.signal)
    print(snr18.noise)
    print(snr18.snr)


.. parsed-literal::

    #################### DER ####################
    Deleting masked values.
    Deleting values of 0.
    Deleting NaN and Inf values.
    100.0
    1.2105394
    82.6078027696
    
