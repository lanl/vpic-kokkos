OAPostprocessing and Visualization
================================

At present, there is no standardized way to visualize or postprocess VPIC data.
Many people do this in their own unique way, and it may be easiest for you to find someone who knows how to do it and ask them nicely to help you.


Visualization with pyvpic
*************************

For simple visualization of field and hydro data, pyvpic_ may be a good option.
Once you have it installed, the following python should work with at least `sample/short_pulse.cxx`::

    >>> import pyvpic.viewer
    >>> pyvpic.viewer.main('global.vpc', interleave=False, order='C')

.. _pyvpic: https://github.com/PrincetonUniversity/pyvpic


Visualization with Paraview
***************************

Paraview can be used for both visualization and post-processing VPIC data.
It can be utilized with a GUI or with python scripting via a module.

Currently, this documentation only covers visualization of VPIC data with a local Paraview install.
However, further documentation on using Paraview for post-processing is in the works.

Connecting local Paraview to server (optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This step is necessary if you are working on a local verison of Paraview and the VPIC data is on a server:

1. Open local Paraview application
2. Click the "Connect" button to connect Paraview to the server
3. Enter the appropriate server configration

Opening VPIC data:
~~~~~~~~~~~~~~~~~~
1. File -> Open. Navigate and select the apprpriate file. Note, when running `sample/short_pulse.cxx` the file to open is `global.vpc`
