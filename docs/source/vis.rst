Postprocessing and Visualization
================================

At present, ParaView is the closest thing to a standardized way to visualize VPIC data.
However, many people do this in their own unique way, and it may be easiest for you to find someone who knows how to do it and ask them nicely to help you.


Visualization with Paraview
***************************

ParaView can be used for both visualization and post-processing VPIC data.
It can be utilized with a GUI or with python scripting via a module.
This is not `ParaView documentation <https://docs.paraview.org/en/latest/index.html>`_.


Connecting local Paraview to server (optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This step is necessary if you are working on a local verison of Paraview and the VPIC data is on a server, e.g., a cluster or supercomputer.

Open your local Paraview application and click the "Connect" button to connect Paraview to the server. Enter the appropriate server configuration and click connect.  The server's administrator should know the appropriate configuration, which is entirely independent of VPIC.

Opening VPIC data
~~~~~~~~~~~~~~~~~
File -> Open. Navigate and select the appropriate VPIC header file. When running `sample/short_pulse.cxx`, this will be `global.vpc` in the run directory.


Visualization with pyvpic
*************************

For simple visualization of field and hydro data, pyvpic_ may be a good option.
Once you have it installed, the following python should work with at least `sample/short_pulse.cxx`::

    >>> import pyvpic.viewer
    >>> pyvpic.viewer.main('global.vpc', interleave=False, order='C')

.. _pyvpic: https://github.com/PrincetonUniversity/pyvpic
