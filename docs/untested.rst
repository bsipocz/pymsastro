
Untested components
===================

- `pymsastro.stats.median_absolute_standard_deviation`
- `pymsastro.image.NDImage`
- `pymsastro.spectrum.synthetic.analyze.AnalyseFactory` (but there is a notebook)
- `pymsastro.stats.reject_minmax` (but there is a notebook)
- ``pymsastro.stats`` after change that they allowed masked values! - though
  it does not seem to be a problem but better check `numpy.ma.median` since
  this returns an array and not a number..
- ``pymsastro.stats`` numerical problems
  (see http://www.johndcook.com/blog/standard_deviation/ ) - though it does not
  seem like a problem.

