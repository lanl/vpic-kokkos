#ifndef _field_advance_private_h_
#define _field_advance_private_h_

#ifndef IN_field_advance
#error "Do not include field_advance_private.h; use field_advance.h"
#endif

#include "field_advance.h"

void
checkpt_field_array_internal( field_array_t * fa );

field_array_t *
restore_field_array_internal( void * params );

#endif // _field_advance_private_h_
