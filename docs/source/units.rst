Units in VPIC
=============

.. *Adapted from the legacy VPIC document* `vpicUnits.tex <https://github.com/lanl/vpic/tree/master/doc/src/vpicUnits.tex>`_
.. *by A. G. Seaton and S. V. Luedtke.*

Units in VPIC can be confusing.
They are user-defined, which can make inheriting a deck confusing for new users.
The reason for this flexibility is that common unit systems, such as SI, are not suitable for use in VPIC because of its use of IEEE 754 single precision floating point numbers.
In single precision, the smallest number representable with full precision is :math:`2^{-126} \approx 1.2\times 10^{-38}`.
As an example, the SI value for :math:`\hbar`, :math:`1.05\times 10^{-34}`, is only four orders of magnitude away from losing precision.
Multiply it by the electron mass, and you are hopelessly lost.

To set up a simulation, the user provides VPIC with values for the time step, cell dimensions, speed of light in vacuum, and vacuum permittivity.
This is either done by directly assigning to the ``grid`` struct or by using the functions ``define_units``, ``define_timestep``, and (for example) ``define_periodic_grid``.
Usually one or more species are defined too, which each require values for the species mass and charge.
These are normally set using the ``define_species`` function.
It is important to understand that VPIC solves the same equations regardless of which values it is given and the chosen unit system.

We will assume here that two unit systems are in use.
The first of these we will refer to as the *user* unit system, and is the user's preferred unit system.
This is often used initially to define parameters of interest to the user, such as the temperature and density of the plasma.
The second we will refer to as the *code* unit system, which is used for values passed to VPIC itself.
This document is concerned primarily with the code unit system.

We will list some common unit systems for values supplied to VPIC later, but for now the primary concerns for choosing a system of units are twofold:

#. The units must be compatible with the SI *equation form* that is solved by VPIC, meaning that Gaussian units cannot be used.
#. Values used by the code (including within calculations) should never exceed :math:`\pm1.7\times 10^{38}`, and a value smaller than :math:`\pm1.2\times 10^{-38}` should not be meaningfully different from zero.


Unit Conversions
****************

In order to calculate values in the code unit system to supply to VPIC, we need to know the unit conversions.
Taking the base units as length, time, mass, and charge, we define conversion factors :math:`L`, :math:`T`, :math:`M`, and :math:`Q` so that a quantity :math:`f_c` in the code unit system is converted to a value :math:`f_u` in the user unit system using :math:`f_u = Ff_c`.

Each value supplied to VPIC can be used to constrain the unit system by writing an equation for its conversion from the user unit system to code units in terms of the base unit conversion factors.
Therefore by picking a subset of these values, we can fully determine the unit system.

To determine what the four base unit conversion factors are, we need four constraining equations.
There are multiple sets of values we could choose.
Given the base units, a straightforward choice would be the timestep, cell size, and species mass and charge.
Here instead we pick the cell size :math:`h`, speed of light :math:`c`, vacuum permittivity :math:`\varepsilon_0`, and the charge-mass ratio :math:`r_{\alpha}` of some species :math:`\alpha`.
This is motivated by many common unit systems setting :math:`\varepsilon_{0,\mathrm{c}} = 1` or :math:`c_{\mathrm{c}} = 1`, as will be seen later.

The constraining equations are then:

.. math::
   :nowrap:

   \begin{align*}
      h_{\mathrm{c}}L                               &= h_{\mathrm{u}},             &
      c_{\mathrm{c}}\frac{L}{T}                     &= c_{\mathrm{u}},             \\
      \varepsilon_{0,\mathrm{c}}\frac{Q^2T^2}{ML^3} &= \varepsilon_{0,\mathrm{u}}, &
      r_{\alpha,\mathrm{c}}\frac{Q}{M}              &= r_{\alpha,\mathrm{u}}.
   \end{align*}

Solving for the conversion factors we find:

.. math::
   :nowrap:

   \begin{align*}
      L &= \frac{h_{\mathrm{u}}}{h_{\mathrm{c}}}, \\
      T &= \frac{h_{\mathrm{u}}}{h_{\mathrm{c}}}\frac{c_{\mathrm{c}}}{c_{\mathrm{u}}}, \\
      M &= \frac{h_{\mathrm{u}}}{h_{\mathrm{c}}}\left(\frac{c_{\mathrm{u}}}{c_{\mathrm{c}}}\right)^2\frac{\varepsilon_{0,\mathrm{u}}}{\varepsilon_{0,\mathrm{c}}}\left(\frac{r_{\alpha,\mathrm{c}}}{r_{\alpha,\mathrm{u}}}\right)^2, \\
      Q &= \frac{h_{\mathrm{u}}}{h_{\mathrm{c}}}\left(\frac{c_{\mathrm{u}}}{c_{\mathrm{c}}}\right)^2\frac{\varepsilon_{0,\mathrm{u}}}{\varepsilon_{0,\mathrm{c}}}\frac{r_{\alpha,\mathrm{c}}}{r_{\alpha,\mathrm{u}}}.
   \end{align*}

These can then be combined into factors for other quantities.
For example, the electric and magnetic field conversion factors :math:`E` and :math:`B` are given by

.. math::
   :nowrap:

   \begin{align*}
      E &= \frac{ML}{QT^2} = \frac{h_{\mathrm{c}}}{h_{\mathrm{u}}}\frac{r_{\alpha,\mathrm{c}}}{r_{\alpha,\mathrm{u}}}\left(\frac{c_{\mathrm{u}}}{c_{\mathrm{c}}}\right)^2, \\
      B &= E\frac{T}{L} = \frac{h_{\mathrm{c}}}{h_{\mathrm{u}}}\frac{r_{\alpha,\mathrm{c}}}{r_{\alpha,\mathrm{u}}}\frac{c_{\mathrm{u}}}{c_{\mathrm{c}}}.
   \end{align*}

Exceptions
----------

In addition to calculating the above conversion factors, the user should be aware of how VPIC normalises variables internally.

Particle Momenta
^^^^^^^^^^^^^^^^

VPIC stores relativistic momentum rather than velocity (which would cause numerical issues) for each particle.
The momentum is stored in dimensionless form given by

.. math::

   p_{\mathrm{norm}} = \frac{p}{m_\alpha c},

where :math:`m_\alpha` is the mass assigned to species :math:`\alpha` and :math:`c` is the speed of light in a vacuum.
This is what must be supplied by the user when performing the particle load.

Magnetic Fields
^^^^^^^^^^^^^^^

For performance reasons, VPIC stores the scaled magnetic field :math:`c\mathbf{B}` rather than :math:`\mathbf{B}`.


Species & Particle Weighting
****************************

To define a particle species in VPIC, the user specifies the species mass :math:`m_{\mathrm{s}}` and charge :math:`q_{\mathrm{s}}`.
Additionally, each particle is individually assigned a weighting factor :math:`w`.
These must be consistent with each other and the above unit system to produce the desired behavior.

Choosing Charge Mass Ratio
--------------------------

The motion of a physical particle with charge :math:`q_{\mathrm{p}}` and mass :math:`m_{\mathrm{p}}` in the EM fields only has explicit dependence on the charge-to-mass ratio :math:`r \equiv q_{\mathrm{p}}/m_{\mathrm{p}}`.
This is also the case for macroparticles in VPIC's particle push.
Therefore the corresponding macroparticle species must be defined such that its charge :math:`q_{\mathrm{s}}` and mass :math:`m_{\mathrm{s}}` satisfy :math:`q_{\mathrm{s}}/m_{\mathrm{s}}=r`.

Choosing Macroparticle Weight
-----------------------------

When calculating the fields generated by a macroparticle, the code uses the species charge and macroparticle weight to calculate a total charge for the macroparticle:

.. math::

   q_{\mathrm{MP}} = wq_{\mathrm{s}}.

For this to result in an amount of current deposition consistent with the desired physical particle density, the macroparticle weight should be defined such that the charge contained in volume :math:`V` is the same for the macroparticles as for physical particles, i.e.

.. math::
   :nowrap:

   \begin{align*}
      q_{\mathrm{p}}n_{\mathrm{p}}V &= q_{\mathrm{MP}}n_{\mathrm{MP}}V \\
                                    &= wq_{\mathrm{s}}n_{\mathrm{MP}}V,
   \end{align*}

where :math:`n_{\mathrm{p}}` and :math:`n_{\mathrm{MP}}` are the physical particle and macroparticle number densities (macroparticles per unit volume) respectively.
So, the particle weight must satisfy

.. math::

   w = \frac{q_{\mathrm{p}}n_{\mathrm{p}}}{q_{\mathrm{s}}n_{\mathrm{MP}}} = \frac{\rho_{\mathrm{p}}}{\rho_{\mathrm{s}}}.

This illustrates that there is a degree of freedom available in that the species charge defined in the deck does not have to be the same as the physical species charge.

Consequences of :math:`q_{\mathrm{s}} \neq q_{\mathrm{p}}`
----------------------------------------------------------

The possibility of :math:`q_{\mathrm{s}} \neq q_{\mathrm{p}}` means that the code has no way of calculating numbers of physical particles since it doesn't store the true particle charge.
This for example is why the hydro dumps contain charge density rather than number density.
Users should therefore also ensure that any custom diagnostics do not assume :math:`q_{\mathrm{s}} = q_{\mathrm{p}}`.


Examples of Common Unit Systems
*******************************

Various choices are often made for the unit system, and we discuss a few of them here.
In general, aside from ensuring that values do not exceed the limits for single-precision floating point numbers, it is often desirable to choose a unit system that eases interpretation of simulation results.
Many of these unit systems allow the governing equations to be written in simpler form, which are convenient in theoretical analysis, and we will illustrate some of these cases here.
As discussed above, the unit system has no bearing on the calculations performed by VPIC -- it always solves the full system of equations.

Electrostatic Plasma Units
--------------------------

In an electrostatic problem simulating an electron plasma, the natural length and time-scales are the Debye length :math:`\lambda_{\mathrm{D}}` and inverse plasma frequency :math:`\omega_{\mathrm{pe}}^{-1}`.
These are also the typical scales for the grid and time-step.
Then choosing the electron charge/mass ratio so that :math:`e/m_e` becomes unity reduces the equation of motion of an electron to

.. math::

   \ddot{\mathbf{r}} = \mathbf{E}.

Finally setting :math:`\varepsilon_0=1` means that Gauss' law is also simplified

.. math::

   \nabla \cdot \mathbf{E} = \rho.

In the deck we would specify

.. math::
   :nowrap:

   \begin{align*}
      h_{\mathrm{c}}             &= \frac{h_{\mathrm{u}}}{\lambda_{\mathrm{D}}}, &
      c_{\mathrm{c}}             &= \frac{c}{v_{\mathrm{th}}},    \\
      \varepsilon_{0,\mathrm{c}} &= 1,                                           &
      r_{e,\mathrm{c}}           &= -1,
   \end{align*}

giving the unit conversions as

.. math::
   :nowrap:

   \begin{align*}
      L &= \lambda_{\mathrm{D}},                                &
      T &= \omega_{\mathrm{pe}}^{-1},                           &
      M &= N_{\mathrm{D}}m_e,                                   \\
      Q &= N_{\mathrm{D}}e,                                     &
      E &= \frac{m_e}{e}v_{\mathrm{th}}\omega_{\mathrm{pe}},    &
      B &= \frac{m_e}{e}\omega_{\mathrm{pe}},
   \end{align*}

where :math:`N_{\mathrm{D}} \equiv n_e\lambda_{\mathrm{D}}^3` is the number of electrons in a "Debye cube".

This unit system allows for some convenient sanity checks.
As mentioned above, the cell size and time-step must be around :math:`1`.
Additionally, if the electron mass in these units is not significantly smaller than 1, then the plasma is not ideal, and simulating it with VPIC is not appropriate.
Interpretation of simulation results may also be more convenient.
For example, the normalized electric field :math:`E_c` is the electron quiver velocity in a plasma wave divided by the thermal velocity -- an important parameter for determining whether wavebreaking will occur.
Similarly, the wavenumber of a plasma wave is normalised such that :math:`k_c = k\lambda_{\mathrm{D}}`, which determines the Landau damping rate.

Electromagnetic Plasma Units
----------------------------

For an electromagnetic problem, the electrostatic units above might be modified so that the length scale becomes the electron skin depth :math:`l_e=c/\omega_{\mathrm{pe}}`.
The Lorentz force and Maxwell's equations become

.. math::

   \ddot{\mathbf{r}} = \mathbf{E} + \mathbf{v} \times \mathbf{B},

and

.. math::
   :nowrap:

   \begin{align*}
      \nabla \times \mathbf{E} &= -\partial_t \mathbf{B}, \\
      \nabla \times \mathbf{B} &= \left(\mathbf{J} + \partial_t \mathbf{E}\right).
   \end{align*}

The deck assignments become:

.. math::
   :nowrap:

   \begin{align*}
      h_{\mathrm{c}}             &= h_{\mathrm{u}}\frac{\omega_{\mathrm{pe}}}{c}, &
      c_{\mathrm{c}}             &= 1, \\
      \varepsilon_{0,\mathrm{c}} &= 1, &
      r_{e,\mathrm{c}}           &= -1,
   \end{align*}

so that the conversion factors are now

.. math::
   :nowrap:

   \begin{align*}
      L &= \frac{c}{\omega_{\mathrm{pe}}},                   &
      T &= \frac{1}{\omega_{\mathrm{pe}}},                   \\
      M &= n_e\frac{c^3}{\omega_{\mathrm{pe}}^3}m_e,         &
      Q &= n_e\frac{c^3}{\omega_{\mathrm{pe}}^3}e,           \\
      E &= \frac{m_e}{e}\omega_{\mathrm{pe}} c,              &
      B &= \frac{m_e}{e}\omega_{\mathrm{pe}}.
   \end{align*}

Note that by removing :math:`\lambda_{\mathrm{D}}` from the unit system, there is no longer any dependence on the electron temperature used to calculate it.

Laser-Plasma Units
------------------

For laser-plasma interaction problems, the unit system can again be modified by replacing the time-scale with the inverse laser frequency, so that the unit system no longer depends on the electron density used to calculate :math:`\omega_{\mathrm{pe}}` and is instead dependent purely on the laser frequency.
In the deck the assignments are

.. math::
   :nowrap:

   \begin{align*}
      h_{\mathrm{c}}             &= h_{\mathrm{u}}k_0, &
      c_{\mathrm{c}}             &= 1,                 \\
      \varepsilon_{0,\mathrm{c}} &= 1,                 &
      r_{e,\mathrm{c}}           &= -1,
   \end{align*}

where :math:`k_0 \equiv \omega_0/c` is the vacuum laser wavenumber.

The conversion factors then become

.. math::
   :nowrap:

   \begin{align*}
      L &= \frac{1}{k_0},                       &
      T &= \frac{1}{\omega_0},                  \\
      M &= \frac{n_{\mathrm{cr}}}{k_0^3}m_e,    &
      Q &= \frac{n_{\mathrm{cr}}}{k_0^3}e,      \\
      E &= \frac{m_e}{e}\omega_0 c,             &
      B &= \frac{m_e}{e}\omega_0,
   \end{align*}

where :math:`n_{\mathrm{cr}} \equiv m_e\varepsilon_0\omega_0^2/e^2` is the critical density.
In this unit system the simulation cell size and time step should satisfy :math:`\Delta x \ll 1` and :math:`\Delta t \ll 1` in order for the field solver to accurately model the laser.
It is also necessary to verify that the cell size resolves the Debye length, which is often a more restrictive condition.
Additionally, the normalized electric (or magnetic) field is the parameter :math:`a_0`.
For :math:`a_0 \ll 1`, this is the electron quiver velocity divided by the speed of light, while :math:`a_0 \gtrsim 1` indicates a relativistic field intensity which will lead to effects such as relativistically induced transparency and relativistic self-focusing.
