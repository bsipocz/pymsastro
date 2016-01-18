# Licensed under a 3-clause BSD style license - see LICENSE.rst

from numpy import mean, median, square, sqrt, sum

from ..utils.decorator_collection import format_doc

__all__ = ['signed_error', 'absolute_error', 'relative_signed_error',
           'relative_absolute_error', 'percentage_signed_error',
           'percentage_absolute_error', 'square_error',
           'square_relative_error', 'mean_signed_error', 'mean_absolute_error',
           'mean_relative_signed_error', 'mean_relative_absolute_error',
           'mean_percentage_absolute_error', 'mean_percentage_signed_error',
           'mean_square_error', 'mean_square_relative_error',
           'root_mean_square_error', 'root_mean_square_relative_error',
           'median_signed_error', 'median_absolute_error',
           'median_relative_absolute_error', 'median_relative_signed_error',
           'median_percentage_absolute_error',
           'median_percentage_signed_error',
           'median_square_error', 'median_square_relative_error',
           'sum_square_error', 'sum_square_relative_error',

           'median_absolute_standard_deviation',

           'error', 'rmse', 'rmse_rel', 'rmsd', 'rmsd_rel', 'mse',
           'residual_sum_of_squares', 'rss']

doc_template = """
Calculates the {0}.

Parameters
----------
value : ``Number`` or `numpy.ndarray`
    The measured value.

reference : ``Number`` or `numpy.ndarray`
    The reference value.
{3}
Returns
-------
{1} : ``Number`` or `numpy.ndarray`
    The {0}.

Notes
-----
The value is calculated by ``{1} = {2}``

{__doc__}
"""

doc_template_kwargs = """

kwargs :
    ``kwargs`` for `numpy.{0}`.
"""


@format_doc(doc_template,
            'signed error',
            'signed_error',
            'value - reference',
            '')
def signed_error(value, reference):
    return value - reference


@format_doc(doc_template,
            'absolute error',
            'abs_error',
            'abs(value - reference)',
            '')
def absolute_error(value, reference):
    return abs(value - reference)


@format_doc(doc_template,
            'relative signed error',
            'rel_signed_error',
            '(value - reference) / abs(reference)',
            '')
def relative_signed_error(value, reference):
    return (value - reference) / abs(reference)


@format_doc(doc_template,
            'relative absolute error',
            'rel_abs_error',
            'abs((value - reference) / reference)',
            '')
def relative_absolute_error(value, reference):
    return abs((value - reference) / reference)


@format_doc(doc_template,
            'percentage signed error',
            'perc_signed_error',
            '100 * (value - reference / abs(reference))',
            '')
def percentage_signed_error(value, reference):
    return 100 * (value - reference) / abs(reference)


@format_doc(doc_template,
            'percentage absolute error',
            'perc_abs_error',
            '100 * (abs(value - reference) / abs(reference))',
            '')
def percentage_absolute_error(value, reference):
    return 100 * abs((value - reference) / reference)


@format_doc(doc_template,
            'square error',
            'square_error',
            'square(value - reference)',
            '')
def square_error(value, reference):
    return square(value - reference)


@format_doc(doc_template,
            'square relative error',
            'square_rel_error',
            'square((value - reference) / reference)',
            '')
def square_relative_error(value, reference):
    return square((value - reference) / reference)


@format_doc(doc_template,
            'mean signed error',
            'mean_signed_error',
            'mean(value - reference)',
            doc_template_kwargs.format('mean'))
def mean_signed_error(value, reference, **kwargs):
    return mean(value - reference, **kwargs)


@format_doc(doc_template,
            'mean absolute error',
            'mean_abs_error',
            'mean(abs(value - reference))',
            doc_template_kwargs.format('mean'))
def mean_absolute_error(value, reference, **kwargs):
    """
References
----------
https://en.wikipedia.org/wiki/Relative_mean_absolute_difference
    """
    return mean(abs(value - reference), **kwargs)


@format_doc(doc_template,
            'mean relative signed error',
            'mean_rel_signed_error',
            'mean((value - reference) / abs(reference))',
            doc_template_kwargs.format('mean'))
def mean_relative_signed_error(value, reference, **kwargs):
    # TODO: Might speed up things if division and absolute were before the mean
    return mean((value - reference) / abs(reference), **kwargs)


@format_doc(doc_template,
            'mean relative absolute error',
            'mean_rel_abs_error',
            'mean(abs(value - reference))',
            doc_template_kwargs.format('mean'))
def mean_relative_absolute_error(value, reference, **kwargs):
    """
References
----------
https://en.wikipedia.org/wiki/Relative_mean_absolute_difference
    """
    return mean(abs((value - reference) / reference), **kwargs)


@format_doc(doc_template,
            'mean percentage signed error',
            'mean_perc_signed_error',
            'mean(100 * (value - reference / abs(reference)))',
            doc_template_kwargs.format('mean'))
def mean_percentage_signed_error(value, reference, **kwargs):
    """
References
----------
https://en.wikipedia.org/wiki/Mean_percentage_error
    """
    return mean(100 * (value - reference) / abs(reference), **kwargs)


@format_doc(doc_template,
            'mean percentage absolute error',
            'mean_perc_abs_error',
            'mean(100 * (abs(value - reference) / abs(reference)))',
            doc_template_kwargs.format('mean'))
def mean_percentage_absolute_error(value, reference, **kwargs):
    """
References
----------
https://en.wikipedia.org/wiki/Mean_percentage_error
    """
    return mean(100 * abs((value - reference) / reference), **kwargs)


@format_doc(doc_template,
            'mean square error',
            'mean_square_error',
            'mean(square(value - reference))',
            doc_template_kwargs.format('mean'))
def mean_square_error(value, reference, **kwargs):
    """
References
----------
https://en.wikipedia.org/wiki/Mean_squared_error
    """
    return mean(square(value - reference), **kwargs)


@format_doc(doc_template,
            'mean square relative error',
            'mean_square_rel_error',
            'mean(square((value - reference) / reference))',
            doc_template_kwargs.format('mean'))
def mean_square_relative_error(value, reference, **kwargs):
    return mean(square((value - reference) / reference), **kwargs)


@format_doc(doc_template,
            'root mean square error',
            'root mean_square_error',
            'sqrt(mean(square(value - reference)))',
            doc_template_kwargs.format('mean'))
def root_mean_square_error(value, reference, **kwargs):
    return sqrt(
                mean(
                     square(
                            value - reference), **kwargs))


@format_doc(doc_template,
            'root mean square relative error',
            'root_mean_square_rel_error',
            'sqrt(mean(square((value - reference) / reference)))',
            doc_template_kwargs.format('mean'))
def root_mean_square_relative_error(value, reference, **kwargs):
    return sqrt(
                mean(
                     square(
                            (value - reference) / reference), **kwargs))


@format_doc(doc_template,
            'median signed error',
            'median_signed_error',
            'median(value - reference)',
            doc_template_kwargs.format('median'))
def median_signed_error(value, reference, **kwargs):
    return median(value - reference, **kwargs)


@format_doc(doc_template,
            'median absolute error',
            'median_abs_error',
            'median(abs(value - reference))',
            doc_template_kwargs.format('median'))
def median_absolute_error(value, reference, **kwargs):
    """
References
----------
https://en.wikipedia.org/wiki/Median_absolute_deviation (not exactly because
this function does not rely on calculating the median of the sample but
allows a reference value/array)
    """
    return median(abs(value - reference), **kwargs)


@format_doc(doc_template,
            'median absolute standard deviation',
            'median_abs_std_dev',
            '1.482602218505602 * median(abs(value - reference))',
            doc_template_kwargs.format('median'))
def median_absolute_standard_deviation(value, reference, **kwargs):
    """
The relation between standard deviation and median absolute deviation requires
a constant scale factor. This factor is approximatly 1.482602218505602.

See also
--------
median_absolute_error
astropy.stats.mad_std

References
----------
https://en.wikipedia.org/wiki/Median_absolute_deviation (not exactly because
this function does not rely on calculating the median of the sample but
allows a reference value/array)
    """
    return 1.482602218505602 * median(abs(value - reference), **kwargs)


@format_doc(doc_template,
            'median relative signed error',
            'median_rel_signed_error',
            'median((value - reference) / abs(reference))',
            doc_template_kwargs.format('median'))
def median_relative_signed_error(value, reference, **kwargs):
    return median((value - reference) / abs(reference), **kwargs)


@format_doc(doc_template,
            'median relative absolute error',
            'median_rel_abs_error',
            'median(abs(value - reference))',
            doc_template_kwargs.format('median'))
def median_relative_absolute_error(value, reference, **kwargs):
    return median(abs((value - reference) / reference), **kwargs)


@format_doc(doc_template,
            'median percentage signed error',
            'median_perc_signed_error',
            'median(100 * (value - reference / abs(reference)))',
            doc_template_kwargs.format('median'))
def median_percentage_signed_error(value, reference, **kwargs):
    return median(100 * (value - reference) / abs(reference), **kwargs)


@format_doc(doc_template,
            'median percentage absolute error',
            'median_perc_abs_error',
            'median(100 * (abs(value - reference) / abs(reference)))',
            doc_template_kwargs.format('median'))
def median_percentage_absolute_error(value, reference, **kwargs):
    return median(100 * abs((value - reference) / reference), **kwargs)


@format_doc(doc_template,
            'median square error',
            'median_square_error',
            'median(square(value - reference))',
            doc_template_kwargs.format('median'))
def median_square_error(value, reference, **kwargs):
    return median(square(value - reference), **kwargs)


@format_doc(doc_template,
            'median square relative error',
            'median_square_rel_error',
            'median(square((value - reference) / reference))',
            doc_template_kwargs.format('median'))
def median_square_relative_error(value, reference, **kwargs):
    return median(square((value - reference) / reference), **kwargs)


@format_doc(doc_template,
            'sum square error',
            'sum_square_error',
            'sum(square(value - reference))',
            doc_template_kwargs.format('sum'))
def sum_square_error(value, reference, **kwargs):
    """
References
----------
https://en.wikipedia.org/wiki/Residual_sum_of_squares
    """
    return sum(square(value - reference), **kwargs)


@format_doc(doc_template,
            'sum square relative error',
            'sum_square_rel_error',
            'sum(square((value - reference) / reference))',
            doc_template_kwargs.format('sum'))
def sum_square_relative_error(value, reference, **kwargs):
    return sum(square((value - reference) / reference), **kwargs)


error = signed_error

rmse = root_mean_square_error
rmse_rel = root_mean_square_relative_error
rmsd = root_mean_square_error
rmsd_rel = root_mean_square_relative_error

mse = mean_square_error

residual_sum_of_squares = sum_square_error
rss = sum_square_error
