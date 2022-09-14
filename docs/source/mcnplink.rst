Linking VPIC to MCNP
============================

`Monte Carlo N-Particle <https://mcnp.lanl.gov/>`_ (MCNP) is a general purpose code for particle transport, and can perform secondary calculations on data produced by VPIC.  This page covers an example of how to prepare VPIC data for MCNP.

Making an MCNP deck
************************************

The example script for making an MCNP deck, `vpic-kokkos/post/VPICtoMCNP.py`, uses data from the :doc:`pbd`, so your VPIC deck must have that enabled. `vpic-kokkos/post/anglehist.c` will bin the particle data into angle-resolved spectra that our script will read.  You may need to reduce the number of angle bins in `anglehist.c` to avoid line-length limits in MCNP.

Once you have the data (electronlostspec.bin) and parameter file (electronanglehistparams.txt), place them in the same directory as VPICtoMCNP.py, and simply run::

    python VPICtoMCNP.py

If you want MCNP to do something other than the default example, you will need to adjust either the python converter or the resulting MCNP deck.  This is not documentation for how to use MCNP.

Running MCNP and plotting the results
**************************************

If you haven't deviated from the example converter, you can run MCNP with something like::

    mcnp6 inp=elecWBrem.mcnp TASKS 36

Again, if you haven't deviated from the example much, you can plot the photon spectra that hits the back of the tungsten converter using `vpic-kokkos/post/mcnpTallySpec.py`:::

    python mcnpTallySpec.py mctal

This plotter looks for the parameter file made by `anglehist.c` to do a normalization.  Note that MCNP can be a bit weird about normalization.  If you have MCNP cut off electrons below 1 MeV (default in this example), but `anglehist.c` put those electrons in a bin, the normalization will be off.  To avoid this, make sure the cut in MCNP is also used in `anglehist.c` (``double Ecut = 1;``).  You might also want to align the cut with one of the bin boundaries to avoid some edge cases.
