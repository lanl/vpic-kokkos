# Vector Particle-In-Cell (VPIC) Project

VPIC is a general-purpose particle-in-cell (PIC) simulation code for modeling multi-species
kinetic plasmas in one, two, or three spatial dimensions.
VPIC can solve equations of motion for different plasma species using kinetic or fluid descriptions.
To solve for the kinetic species, VPIC employs a variety of explicit and implicit time-stepping
schemes to update charged particle positions and velocities.
In contrast, the fluid species is advanced in time by solving a number of fluid moment equations on
a spatial grid using finite-difference methods.
The electromagnetic fields are also solved on a spatial grid using either the full Maxwell equations,
or various approximations to the Maxwell equations that target particular temporal and spatial scales
and physics approximations such as a low frequency quasi-neutral approximation that uses an Ohm's law
for the electric field.
The kinetic particle quantities are coupled with the grid-based fields and fluid moments by giving the
particles a shape of selectable order.
This allows the fields to be interpolated from spatial grid points to the particle positions, and current
densities to be collected from the particles onto the spatial grid points.

# Attribution

Researchers who use the VPIC code for scientific research are asked to cite
the papers by Kevin Bowers listed below.

1. Le, A., Winske, D., Stanier, A., Daughton, W., Cowee, M., Wetherton, B.,
& Guo, F. (2021). Astrophysical explosions revisited: collisionless coupling
of debris to magnetized plasma. Journal of Geophysical Research: Space Physics,
126(9), e2021JA029125.

2. K.J. Bowers, B.J. Albright, B. Bergen and T.J.T. Kwan, Ultrahigh
performance three-dimensional electromagnetic relativistic kinetic
plasma simulation, Phys. Plasmas 15, 055703 (2008);
http://dx.doi.org/10.1063/1.2840133

# Getting the Code

VPIC uses nested submodules.  This requires the addition of the *--recursive*
flag when cloning the repository:

    % git clone https://github.com/lanl/vpic-kokkos.git
    % git checkout hybridVPIC 

This command will check out the VPIC source code.

# Requirements

The primary requirement to build VPIC is a C++11 capable compiler and
an up-to-date version of MPI.

# Build Instructions

    % cd vpic-kokkos 

VPIC uses the CMake build system. To configure a build, do the following from
the top-level source directory:
  
    % mkdir build
    % cd build

Then call the curses version of CMake:

    % ccmake ..

The `./arch` directory also contains various cmake scripts (including specific build options) which can help with building

They can be invoked using something like:

    % ../arch/generic-Release

GCC users should ensure the `-fno-strict-aliasing` compiler flag is set (as shown in `./arch/generic-gcc-sse`)

After configuration, simply type 'make'.

# Building an example input deck

After you have successfully built VPIC, you should have an executable in
the *bin* directory called *vpic*.  To build an executable from one of
the example input decks you can simply run:

    % bin/vpic input_deck

or change the PROJECTDIR variable in the example/.. Makefiles to

PROJECTDIR = vpic-kokkos/build/bin

Beginners are advised to read the example decks, as they provide many examples of common uses cases.

# Command Line Arguments

Note: Historic VPIC users should note that the format of command line arguments was changed in the first open source release. The equals symbol is no longer accepted, and two dashes are mandatory. 

In general, command line arguments take the form `--command value`, in which two dashes are followed by a keyword, with a space delimiting the command and the value.

The following specific syntax is available to the users:

## Threading

Threading (per MPI rank) can be enabled using the following syntax: 

`./binary.Linux --tpp n`

Where n specifies the number of threads

### Example:

`mpirun -n 2 ./binary.Linux --tpp 2`

To run with VPIC with two threads per MPI rank.

## Checkpoint Restart

VPIC can restart from a checkpoint dump file, using the following syntax:

`./binary.Linux --restore <path to file>`

### Example:

`./binary.Linux --restore ./restart/restart0`

To restart VPIC using the restart file `./restart/restart0`

# Feedback

Feedback, comments, or issues can be raised through [GitHub issues](https://github.com/lanl/vpic-kokkos/issues)

# Release

This software has been approved for open source release and has been assigned **LA-CC-15-109**.

# Copyright

© 2022. Triad National Security, LLC. All rights reserved. This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.

This program is open source under the BSD-3 License. Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

    Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

    Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

    Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.