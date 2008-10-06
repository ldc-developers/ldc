
/*
 * Placed in public domain.
 * Written by Hauke Duden and Walter Bright
 */

/* This is for use with variable argument lists with extern(D) linkage. */

/* Modified for LDC (LLVM D Compiler) by Tomas Lindquist Olsen, 2007 */

module std.stdarg;

alias void* va_list;

T va_arg(T)(ref va_list vp)
{
    size_t size = T.sizeof > size_t.sizeof ? size_t.sizeof : T.sizeof;
    va_list vptmp = cast(va_list)((cast(size_t)vp + size - 1) &  ~(size - 1));
    vp = vptmp + T.sizeof;
    return *cast(T*)vptmp;
}
