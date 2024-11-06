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


The primary documentation for VPIC has moved to Sphinx and is hosted on GitHub
Pages [here](https://lanl.github.io/vpic-kokkos/index.html), and located in
`docs/`.  The documentation is still a work in progress, but hopefully
sufficient to get most users started.

# Attribution

Researchers who use the hybridVPIC-K code for scientific research are asked to cite
the papers listed below.

1. Le, A., Winske, D., Stanier, A., Daughton, W., Cowee, M., Wetherton, B.,
& Guo, F. (2021). Astrophysical explosions revisited: collisionless coupling
of debris to magnetized plasma. Journal of Geophysical Research: Space Physics,
126(9), e2021JA029125.

2. Bird, R., Tan, N., Luedtke, S. V., Harrell, S. L., Taufer, M., & Albright,
B. (2021). VPIC 2.0: Next generation particle-in-cell simulations. IEEE
Transactions on Parallel and Distributed Systems, 33(4), 952-963.

# Getting the Code

VPIC uses nested submodules.  This requires the addition of the *--recursive*
flag when cloning the repository:

    % git clone --recurse-submodules https://github.com/lanl/vpic-kokkos.git
    % git checkout hybridVPIC-K 

This command will check out the VPIC source code.

# Copyright

© 2022. Triad National Security, LLC. All rights reserved.  This program was
produced under U.S. Government contract 89233218CNA000001 for Los Alamos
National Laboratory (LANL), which is operated by Triad National Security, LLC
for the U.S.  Department of Energy/National Nuclear Security Administration.
All rights in the program are reserved by Triad National Security, LLC, and the
U.S. Department of Energy/National Nuclear Security Administration. The
Government is granted for itself and others acting on its behalf a
nonexclusive, paid-up, irrevocable worldwide license in this material to
reproduce, prepare derivative works, distribute copies to the public, perform
publicly and display publicly, and to permit others to do so.

This program is open source under the BSD-3 License.  Redistribution and use in
source and binary forms, with or without modification, are permitted provided
that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
 
2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.
 
3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
