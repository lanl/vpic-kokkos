## Core Features                                                                                                                                                

### User Initialization [SUPPORTED]

Full supported and unchanged. Initialization happens on the CPU, before data is moved down to the GPU.                                                                                

### Particle Sorting [SUPPORTED]

Fully support, as an out of place sort                                                                                                                          

### User Particle Injection [SLOW]

Supported, but requires a full data copy                                                                                                                        

### User Current Injection [SLOW]

Supported, but requires a full data copy                                                                                                                           

### User Field Injection [SLOW]

Supported, but requires a full data copy                                                                                                                              

### Field Cleaning (B+E) [SUPPORTED]

Fully supported                                                                                                                                                       

### User Diagnostics [SUPPORTED]

Supported in two ways:

- GPU: If a user writes GPU-aware code in their deck, it will work and be fast
- CPU: A user can use existing diagnostics by requesting the code to ship the data back to the CPU, at the expense of speed (see "New Features")                                                                                
### Collisions [SUPPORTED]

Basic TA support added, but not heavily tested.                                                                                                                       

### User Boundary Conditions [NOT SUPPORTED]

Needs manual porting                                       

### User Particle Collisions [NOT SUPPORTED]

Only the inbuilt TA collisions are supported at this time                                                                                                       

### Emitter List [NOT SUPPORTED]

Not used by any major uses. Will be ported on request                                                                                                           

## New Features                                                                                                                                                 

### User specified Data Copy Intervals [SLOW]

Users can request that particle or field data be copied back to the host via intervals set in the input deck. This copy happens before user diagnostics to allow the use of existing diagnostics code at the expense of performance
