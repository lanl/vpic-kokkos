Welcome to the Kokkos version of the Vector Particle-in-Cell code, VPIC 2.0!
VPIC is a 3D3V, fully relativistic, kinetic, performance-first PIC code for
solving the coupled Maxwell-Boltzmann system of equations. Utilizing the Kokkos
performance-portable framework, VPIC achieves high performance on multiple CPU
and GPU architectures.

The primary documentation for VPIC has moved to Sphinx and is hosted on GitHub
Pages [here](https://lanl.github.io/vpic-kokkos/index.html), and located in
`docs/`.  The documentation is still a work in progress, but hopefully
sufficient to get most users started.

# Attribution

Researchers who use the VPIC code for scientific research are asked to cite
the papers listed below.

1. Bird, R., Tan, N., Luedtke, S. V., Harrell, S. L., Taufer, M., & Albright,
B. (2021). VPIC 2.0: Next generation particle-in-cell simulations. IEEE
Transactions on Parallel and Distributed Systems, 33(4), 952-963.

2. Bowers, K. J., B. J. Albright, B. Bergen, L. Yin, K. J. Barker and
D. J. Kerbyson, "0.374 Pflop/s Trillion-Particle Kinetic Modeling of
Laser Plasma Interaction on Road-runner," Proc. 2008 ACM/IEEE Conf.
Supercomputing (Gordon Bell Prize Finalist Paper).
http://dl.acm.org/citation.cfm?id=1413435

3. K.J. Bowers, B.J. Albright, B. Bergen and T.J.T. Kwan, Ultrahigh
performance three-dimensional electromagnetic relativistic kinetic
plasma simulation, Phys. Plasmas 15, 055703 (2008);
http://dx.doi.org/10.1063/1.2840133

4. K.J. Bowers, B.J. Albright, L. Yin, W. Daughton, V. Roytershteyn,
B. Bergen and T.J.T Kwan, Advances in petascale kinetic simulations
with VPIC and Roadrunner, Journal of Physics: Conference Series 180,
012055, 2009


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
