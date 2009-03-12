#ifndef __RMEM_H__
#define __RMEM_H__

// jam memory stuff here

#include "mem.h"

#if (defined (__SVR4) && defined (__sun))
#include <alloca.h>
#endif

#ifdef __MINGW32__
#include <malloc.h>
#endif

#endif // __RMEM_H__
