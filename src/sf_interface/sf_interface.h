#ifndef _sf_interface_h_
#define _sf_interface_h_

// FIXME: THE HOST PROCESSED FIELD KERNELS SHOULD BE UPDATED TO USE
// SCALAR FMA INSTRUCTIONS WITH COMMENSURATE ROUND-OFF PROPERTIES TO
// THE FMA INSTRUCTIONS USED ON THE PIPELINE PROCESSED FIELDS!

// FIXME: (nx>1) ? (1/dx) : 0 TYPE LOGIC SHOULD BE FIXED SO THAT NX
// REFERS TO THE GLOBAL NUMBER OF CELLS IN THE X-DIRECTION (NOT THE
// _LOCAL_ NUMBER OF CELLS).  THIS LATENT BUG IS NOT EXPECTED TO
// AFFECT ANY PRACTICAL SIMULATIONS.

#include "../field_advance/field_advance.h"

#include "hydro/hydro_array.h"
#include "interpolator/interpolator_array.h"
#include "accumulator/accumulator_array.h"

// FIXME: SHOULD INCLUDE SPECIES_ADVANCE TOO ONCE READY



#endif // _sf_interface_h_
