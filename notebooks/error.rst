
Error Measurements
==================

There are several different methods to calculate the error of a
measurement om comparison to a theoretical (or approximated value):

General Error
-------------

-  "signed error" (also "error"):

   .. math:: measurement - theoretical

-  "absolute error":

   .. math:: abs(measurement - theoretical)

Relative Error
--------------

based on these there are several other quantities (they are possible for
signed and absolute errors)

-  "relative error":

   .. math:: error \div theoretical

-  "percentage error":

   .. math:: 100 \times error \div theoretical

Squared Error
-------------

and several more advanced error definitions (based on square):

-  "square\_error", "square\_relative\_error"
-  "mean\_square\_error" (also "mse"), "mean\_square\_relative\_error"
-  "root\_mean\_square\_error" (also "rmse"),
   "root\_mean\_square\_relative\_error" (also "rmse\_rel")
-  "median\_square\_error", "median\_square\_relative\_error"
-  "sum\_square\_error" (also avaible under "residual\_sum\_of\_squares"
   or "rss"), "sum\_square\_relative\_error"

Mean Error
----------

-  "mean\_signed\_error", "mean\_absolute\_error"
-  "mean\_relative\_signed\_error", "mean\_relative\_absolute\_error"
-  "mean\_percentage\_signed\_error", mean\_percentage\_absolute\_error"

Median Error
------------

-  "median\_signed\_error", "median\_absolute\_error"
-  "median\_relative\_signed\_error",
   "median\_relative\_absolute\_error"
-  "median\_percentage\_signed\_error",
   "median\_percentage\_absolute\_error"

.. code:: python

    from error import *

For simplicity I'll create a function that will call all the error
functions and displays the result at the end

.. code:: python

    # For simplicity I'll make a function that calls each error calculation function and prints the result
    def make_error_computation(measured, theory):
        error = [signed_error, absolute_error,
                 relative_signed_error, relative_absolute_error,
                 percentage_signed_error, percentage_absolute_error,
                 square_error, square_relative_error,
                 mean_square_error, mean_square_relative_error,
                 root_mean_square_error, root_mean_square_relative_error,
                 median_square_error, median_square_relative_error,
                 sum_square_error, sum_square_relative_error,
                 mean_signed_error, mean_absolute_error,
                 mean_relative_signed_error, mean_relative_absolute_error,
                 mean_percentage_signed_error, mean_percentage_absolute_error,
                 median_signed_error, median_absolute_error,
                 median_relative_signed_error, median_relative_absolute_error,
                 median_percentage_signed_error, median_percentage_absolute_error]
    
        for i in error:
            print("{0}: {1}".format(i.__name__, i(measured, theory)))

Now let's have some try. Suppose we have a measurement of something that
has a theoretical value of 100. The value is measured once with 101.

.. code:: python

    theory = 100
    measurement = 101
    
    make_error_computation(measurement, theory)


.. parsed-literal::

    signed_error: 1
    absolute_error: 1
    relative_signed_error: 0.01
    relative_absolute_error: 0.01
    percentage_signed_error: 1.0
    percentage_absolute_error: 1.0
    square_error: 1
    square_relative_error: 0.0001
    mean_square_error: 1.0
    mean_square_relative_error: 0.0001
    root_mean_square_error: 1.0
    root_mean_square_relative_error: 0.01
    median_square_error: 1.0
    median_square_relative_error: 0.0001
    sum_square_error: 1
    sum_square_relative_error: 0.0001
    mean_signed_error: 1.0
    mean_absolute_error: 1.0
    mean_relative_signed_error: 0.01
    mean_relative_absolute_error: 0.01
    mean_percentage_signed_error: 1.0
    mean_percentage_absolute_error: 1.0
    median_signed_error: 1.0
    median_absolute_error: 1.0
    median_relative_signed_error: 0.01
    median_relative_absolute_error: 0.01
    median_percentage_signed_error: 1.0
    median_percentage_absolute_error: 1.0
    

Okay most of them are the same and very little happens here. So we
measure the value again and this time it is 98.

.. code:: python

    theory = 100
    measurement = 98
    
    make_error_computation(measurement, theory)


.. parsed-literal::

    signed_error: -2
    absolute_error: 2
    relative_signed_error: -0.02
    relative_absolute_error: 0.02
    percentage_signed_error: -2.0
    percentage_absolute_error: 2.0
    square_error: 4
    square_relative_error: 0.0004
    mean_square_error: 4.0
    mean_square_relative_error: 0.0004
    root_mean_square_error: 2.0
    root_mean_square_relative_error: 0.02
    median_square_error: 4.0
    median_square_relative_error: 0.0004
    sum_square_error: 4
    sum_square_relative_error: 0.0004
    mean_signed_error: -2.0
    mean_absolute_error: 2.0
    mean_relative_signed_error: -0.02
    mean_relative_absolute_error: 0.02
    mean_percentage_signed_error: -2.0
    mean_percentage_absolute_error: 2.0
    median_signed_error: -2.0
    median_absolute_error: 2.0
    median_relative_signed_error: -0.02
    median_relative_absolute_error: 0.02
    median_percentage_signed_error: -2.0
    median_percentage_absolute_error: 2.0
    

At least this time the difference between signed and absolute errors is
visible but it is still mostly the same value. But what happens if we
include both measured values?

I'll use a numpy array to include both measurements.

.. code:: python

    import numpy as np
    theory = 100
    measurement = np.array([101, 98])
    
    make_error_computation(measurement, theory)


.. parsed-literal::

    signed_error: [ 1 -2]
    absolute_error: [1 2]
    relative_signed_error: [ 0.01 -0.02]
    relative_absolute_error: [ 0.01  0.02]
    percentage_signed_error: [ 1. -2.]
    percentage_absolute_error: [ 1.  2.]
    square_error: [1 4]
    square_relative_error: [ 0.0001  0.0004]
    mean_square_error: 2.5
    mean_square_relative_error: 0.00025
    root_mean_square_error: 1.5811388300841898
    root_mean_square_relative_error: 0.015811388300841896
    median_square_error: 2.5
    median_square_relative_error: 0.00025
    sum_square_error: 5
    sum_square_relative_error: 0.0005
    mean_signed_error: -0.5
    mean_absolute_error: 1.5
    mean_relative_signed_error: -0.005
    mean_relative_absolute_error: 0.015
    mean_percentage_signed_error: -0.5
    mean_percentage_absolute_error: 1.5
    median_signed_error: -0.5
    median_absolute_error: 1.5
    median_relative_signed_error: -0.005
    median_relative_absolute_error: 0.015
    median_percentage_signed_error: -0.5
    median_percentage_absolute_error: 1.5
    

Okay that's more like a statistic. But since we have the experiment up
and running we will start measuring the value much more often. Suppose a
million times.

Instead of typing all different measurements I'll use the pseudo-random
number generator of numpy with a normal distributed value of 100 and a
standard deviation of 3 and we'll measure it a million times.

.. code:: python

    theory = 100
    measurement = np.random.normal(100, 3, 1000000)
    
    make_error_computation(measurement, theory)


.. parsed-literal::

    signed_error: [-1.40984237  0.10079109 -2.85309461 ..., -0.33546659 -0.54275037
      0.37003034]
    absolute_error: [ 1.40984237  0.10079109  2.85309461 ...,  0.33546659  0.54275037
      0.37003034]
    relative_signed_error: [-0.01409842  0.00100791 -0.02853095 ..., -0.00335467 -0.0054275   0.0037003 ]
    relative_absolute_error: [ 0.01409842  0.00100791  0.02853095 ...,  0.00335467  0.0054275   0.0037003 ]
    percentage_signed_error: [-1.40984237  0.10079109 -2.85309461 ..., -0.33546659 -0.54275037
      0.37003034]
    percentage_absolute_error: [ 1.40984237  0.10079109  2.85309461 ...,  0.33546659  0.54275037
      0.37003034]
    square_error: [ 1.98765551  0.01015884  8.14014886 ...,  0.11253783  0.29457796
      0.13692245]
    square_relative_error: [  1.98765551e-04   1.01588448e-06   8.14014886e-04 ...,   1.12537830e-05
       2.94577964e-05   1.36922452e-05]
    mean_square_error: 9.003233383782348
    mean_square_relative_error: 0.0009003233383782347
    root_mean_square_error: 3.0005388489040343
    root_mean_square_relative_error: 0.030005388489040344
    median_square_error: 4.087575501005665
    median_square_relative_error: 0.0004087575501005664
    sum_square_error: 9003233.383782348
    sum_square_relative_error: 900.3233383782347
    mean_signed_error: 0.0027074070970562595
    mean_absolute_error: 2.393772346974627
    mean_relative_signed_error: 2.7074070970562596e-05
    mean_relative_absolute_error: 0.02393772346974627
    mean_percentage_signed_error: 0.0027074070970562595
    mean_percentage_absolute_error: 2.393772346974627
    median_signed_error: -0.00015840667219180204
    median_absolute_error: 2.02177533395232
    median_relative_signed_error: -1.5840667219180203e-06
    median_relative_absolute_error: 0.0202177533395232
    median_percentage_signed_error: -0.00015840667219180204
    median_percentage_absolute_error: 2.02177533395232
    

Now this is more like a statistic and some quantities are very handy to
have.

-  The "root\_mean\_square\_error" is almost 3. Which is just the
   standard deviation of our random number generator. So this can be
   used as a measurement for the standard-deviation.

-  The "mean\_square\_error" is the square of the
   "root\_mean\_square\_error" and since we identified the latter as
   standard-deviation approximation the square of it must be the
   variance. A difference between the mean\_squared\_error and the
   variance would be identified as the BIAS of the measurement.

-  Apart from the "root\_mean\_square\_error" the
   "median\_absolute\_error" is also an approximator for the standard
   deviation but since the median has some other characteristics than
   the mean we have to multiply it by approximatly 1.4826 (see
   https://en.wikipedia.org/wiki/Median\_absolute\_deviation) for an
   explanation.

-  The "mean\_absolute\_difference" is a measurement for the statistical
   dispersion.

-  The "mean\_signed\_error" and "median\_signed\_error" are a
   measurement for the goodness of the measured value to the arithmetic
   mean or median of the sample. A value of zero or close to zero means
   that the mean/median of the sample is very similar to the theoretical
   value.

-  The "mean\_percentage\_error" gives the average percentage of
   difference for each measurement.

-  Another important quantity is the
   "root\_mean\_square\_relative\_error" since the inverse of it would
   give the *signal-to-noise ratio*

But another example where most statistic packages I know give strange
results: If the theoretical value is negative. It will not happen often
(rather very rare) but these error measurements can cope with it.

Suppose now we have a theoretical value of -100 and measure -95

.. code:: python

    theory = -100
    measurement = -95
    
    make_error_computation(measurement, theory)


.. parsed-literal::

    signed_error: 5
    absolute_error: 5
    relative_signed_error: 0.05
    relative_absolute_error: 0.05
    percentage_signed_error: 5.0
    percentage_absolute_error: 5.0
    square_error: 25
    square_relative_error: 0.0025000000000000005
    mean_square_error: 25.0
    mean_square_relative_error: 0.0025000000000000005
    root_mean_square_error: 5.0
    root_mean_square_relative_error: 0.05
    median_square_error: 25.0
    median_square_relative_error: 0.0025000000000000005
    sum_square_error: 25
    sum_square_relative_error: 0.0025000000000000005
    mean_signed_error: 5.0
    mean_absolute_error: 5.0
    mean_relative_signed_error: 0.05
    mean_relative_absolute_error: 0.05
    mean_percentage_signed_error: 5.0
    mean_percentage_absolute_error: 5.0
    median_signed_error: 5.0
    median_absolute_error: 5.0
    median_relative_signed_error: 0.05
    median_relative_absolute_error: 0.05
    median_percentage_signed_error: 5.0
    median_percentage_absolute_error: 5.0
    

Works as expected. Just to be on the safe side we measure -102 for the
same quantity.

.. code:: python

    theory = -100
    measurement = -102
    
    make_error_computation(measurement, theory)


.. parsed-literal::

    signed_error: -2
    absolute_error: 2
    relative_signed_error: -0.02
    relative_absolute_error: 0.02
    percentage_signed_error: -2.0
    percentage_absolute_error: 2.0
    square_error: 4
    square_relative_error: 0.0004
    mean_square_error: 4.0
    mean_square_relative_error: 0.0004
    root_mean_square_error: 2.0
    root_mean_square_relative_error: 0.02
    median_square_error: 4.0
    median_square_relative_error: 0.0004
    sum_square_error: 4
    sum_square_relative_error: 0.0004
    mean_signed_error: -2.0
    mean_absolute_error: 2.0
    mean_relative_signed_error: -0.02
    mean_relative_absolute_error: 0.02
    mean_percentage_signed_error: -2.0
    mean_percentage_absolute_error: 2.0
    median_signed_error: -2.0
    median_absolute_error: 2.0
    median_relative_signed_error: -0.02
    median_relative_absolute_error: 0.02
    median_percentage_signed_error: -2.0
    median_percentage_absolute_error: 2.0
    

It works even if you have a theoretical negative value and measure a
positive value (or vise-versa)

.. code:: python

    theory = -2
    measurement = 1
    
    make_error_computation(measurement, theory)


.. parsed-literal::

    signed_error: 3
    absolute_error: 3
    relative_signed_error: 1.5
    relative_absolute_error: 1.5
    percentage_signed_error: 150.0
    percentage_absolute_error: 150.0
    square_error: 9
    square_relative_error: 2.25
    mean_square_error: 9.0
    mean_square_relative_error: 2.25
    root_mean_square_error: 3.0
    root_mean_square_relative_error: 1.5
    median_square_error: 9.0
    median_square_relative_error: 2.25
    sum_square_error: 9
    sum_square_relative_error: 2.25
    mean_signed_error: 3.0
    mean_absolute_error: 3.0
    mean_relative_signed_error: 1.5
    mean_relative_absolute_error: 1.5
    mean_percentage_signed_error: 150.0
    mean_percentage_absolute_error: 150.0
    median_signed_error: 3.0
    median_absolute_error: 3.0
    median_relative_signed_error: 1.5
    median_relative_absolute_error: 1.5
    median_percentage_signed_error: 150.0
    median_percentage_absolute_error: 150.0
    

Other measurement functions
===========================

There are some functions included that are not strictly about errors.
These include:

-  "root\_mean\_square" (also "rms"): The RMS of an array
-  "sum\_square" (also "ss", "sum\_of\_squares"): The sum of the squares
   of an array
-  "root\_sum\_square": The square root of the sum\_square

Also different "mean" methods are imported from numpy, scipy:
=============================================================

-  "arithmetic\_mean": "numpy.mean"
-  "quadratic\_mean": "root\_mean\_square"
-  "harmonic\_mean": "scipy.stats.hmean"
-  "geometric\_mean": "scipy.stats.gmean"

For simplicity I'll make a function that calls each calculation function
and prints the result like I did with the errors.

.. code:: python

    def make_other_computation(measured):
        other = [root_mean_square, sum_square, root_sum_square,
                 arithmetic_mean, quadratic_mean, harmonic_mean, geometric_mean]
    
        for i in other:
            try:
                print("{0}: {1}".format(i.__name__, i(measured)))
            except IndexError:
                print("{0} cannot be used with this input.".format(i.__name__))

Since part of these functions are defined by scipy and numpy they might
not accept single values:

.. code:: python

    measurement = 100
    make_other_computation(measurement)


.. parsed-literal::

    root_mean_square: 100.0
    sum_square: 10000
    root_sum_square: 100.0
    mean: 100.0
    root_mean_square: 100.0
    hmean cannot be used with this input.
    gmean cannot be used with this input.
    

But all of these work with arrays

.. code:: python

    measurement = np.array([1,2,3])
    make_other_computation(measurement)


.. parsed-literal::

    root_mean_square: 2.160246899469287
    sum_square: 14
    root_sum_square: 3.7416573867739413
    mean: 2.0
    root_mean_square: 2.160246899469287
    hmean: 1.6363636363636365
    gmean: 1.8171205928321397
    

and again with a different array:

.. code:: python

    measurement = np.array([1,2,3,3,2,1])
    make_other_computation(measurement)


.. parsed-literal::

    root_mean_square: 2.160246899469287
    sum_square: 28
    root_sum_square: 5.291502622129181
    mean: 2.0
    root_mean_square: 2.160246899469287
    hmean: 1.6363636363636365
    gmean: 1.8171205928321397
    

Benchmarks
==========

Python functions cause overhead so for single elements it slows down the
code (up to 15x slower) Also the functions are optimized to allow for
negative theoretical values and more, so sometimes there are unnecessary
calls to abs() which could be avoided if you can exclude that
theoretical values could be negative.

But as speed does not always matter, sometimes having more descriptive
names is better in understanding the code you have written years ago.

So using these functions is limited to cases where: - speed does not
matter - a more descriptive function is more comprehensable than just
writing the operation - arrays are used.

-  you are not always sure where the absolutes have to be (like me) ...
   :-) but I'm not the only one
   http://mathworld.wolfram.com/RelativeDeviation.html gives the
   relative deviation as "abs(measured-theory)/theory" but it should be
   "abs((measured-theory)/theory)" if we allow theoretical values below
   0... not quite often but it happens.

Signed Error Computation difference
-----------------------------------

Error function is 7-8 times slower.

.. code:: python

    %timeit 100-98
    %timeit signed_error(100,98)
    assert 100-98 == signed_error(100,98)


.. parsed-literal::

    The slowest run took 18.39 times longer than the fastest. This could mean that an intermediate result is being cached 
    10000000 loops, best of 3: 60.7 ns per loop
    The slowest run took 7.71 times longer than the fastest. This could mean that an intermediate result is being cached 
    1000000 loops, best of 3: 434 ns per loop
    

Relative Signed Error Computation difference
--------------------------------------------

Error function is 10 times slower.

.. code:: python

    %timeit (98-100)/100
    %timeit relative_signed_error(98, 100)
    import numpy as np
    assert (98-100)/100 == relative_signed_error(98, 100)


.. parsed-literal::

    10000000 loops, best of 3: 60.8 ns per loop
    The slowest run took 7.89 times longer than the fastest. This could mean that an intermediate result is being cached 
    1000000 loops, best of 3: 708 ns per loop
    

Percentage Signed Error Computation difference
----------------------------------------------

Error function is 15 times slower.

.. code:: python

    %timeit 100*(98-100)/100
    %timeit percentage_signed_error(98, 100)
    assert 100*(98-100)/100 == percentage_signed_error(98, 100)


.. parsed-literal::

    10000000 loops, best of 3: 60.8 ns per loop
    The slowest run took 7.22 times longer than the fastest. This could mean that an intermediate result is being cached 
    1000000 loops, best of 3: 851 ns per loop
    

Percentage Signed Error Computation difference with Arrays
----------------------------------------------------------

Error function is only a little slower (even for a 1-element array) due
to numpy array overhead.

.. code:: python

    a = np.ones(1) * 98
    b = np.ones(1) * 100
    %timeit 100*(a-b)/b
    %timeit percentage_signed_error(a, b)


.. parsed-literal::

    100000 loops, best of 3: 15.4 µs per loop
    10000 loops, best of 3: 20.6 µs per loop
    
