/**
 * Shims for libunwind macros on ARM.
 *
 * It would be possible to reimplement those entirely in D, but to avoid
 * an unmaintainable amount of dependencies on internal implementation details,
 * we use the C versions instead.
 *
 * Copyright: David Nadlinger, 2012.
 * License:   <a href="http://www.boost.org/LICENSE_1_0.txt">Boost License 1.0</a>.
 * Authors:   David Nadlinger
 */

#ifdef __ARM_EABI__

#include <limits.h> // for __GLIBC__
#include <unwind.h>

// clang's unwind.h doesn't have this
typedef struct _Unwind_Context _Unwind_Context;

_Unwind_Ptr _d_eh_GetIP(_Unwind_Context *context)
{
    return _Unwind_GetIP(context);
}

#ifdef __GLIBC__
_Unwind_Ptr _d_eh_GetIPInfo(_Unwind_Context *context, int *ptr)
{
    return _Unwind_GetIPInfo(context, ptr);
}
#endif

void _d_eh_SetIP(_Unwind_Context *context, _Unwind_Ptr new_value)
{
    _Unwind_SetIP(context, new_value);
}

_Unwind_Word _d_eh_GetGR(_Unwind_Context *context, int index)
{
    return _Unwind_GetGR(context, index);
}

void _d_eh_SetGR(_Unwind_Context *context, int index, _Unwind_Word new_value)
{
    _Unwind_SetGR(context, index, new_value);
}

#endif
