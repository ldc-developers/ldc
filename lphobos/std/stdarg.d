
/*
 * Placed in public domain.
 * Written by Hauke Duden and Walter Bright
 */

/* This is for use with variable argument lists with extern(D) linkage. */

module std.stdarg;

alias void* va_list;

T va_arg(T)(inout va_list vp)
{
    va_list vptmp = vp;
    vp += T.sizeof;
    return *cast(T*)vptmp;
}
