
/*
 * Placed in public domain.
 * Written by Hauke Duden and Walter Bright
 */

/* This is for use with variable argument lists with extern(D) linkage. */

module std.stdarg;

alias void* va_list;

T va_arg(T)(inout va_list vp)
{
    static assert((T.sizeof & (T.sizeof -1)) == 0);
    va_list vptmp = cast(va_list)((cast(size_t)vp + T.sizeof - 1) &  ~(T.sizeof - 1));
    vp = vptmp + T.sizeof;
    return *cast(T*)vptmp;
}
