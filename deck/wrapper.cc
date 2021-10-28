#include "wrapper.h"

// Include the users input deck
#define EMPTY()
#define shallow(s) s EMPTY()
#include EXPAND_AND_STRINGIFY(shallow(INPUT_DECK))
