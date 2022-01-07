Postprocessing and Visualization
================================

At present, there is no standardized way to visualize or postprocess VPIC data.
Many people do this in their own unique way, and it may be easiest for you to find someone who knows how to do it and ask them nicely to help you.

For simple visualization of field and hydro data, pyvpic_ may be a good option.
Once you have it installed, the following python should work with at least `sample/short_pulse.cxx`::

    >>> import pyvpic.viewer
    >>> pyvpic.viewer.main('global.vpc', interleave=False, order='C')

.. _pyvpic: https://github.com/PrincetonUniversity/pyvpic
