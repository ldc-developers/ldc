
/**
 * C's &lt;stdarg.h&gt;
 * Authors: Hauke Duden, Walter Bright and Tomas Lindquist Olsen, Digital Mars, www.digitalmars.com
 * License: Public Domain
 * Macros:
 *	WIKI=Phobos/StdCStdarg
 */

/* This is for use with extern(C) variable argument lists. */

module std.c.stdarg;

public import llvm.va_list;

pragma(va_start)
    void va_start(T)(va_list ap, ref T);

pragma(va_arg)
    T va_arg(T)(va_list ap);

pragma(va_end)
    void va_end(va_list args);

pragma(va_copy)
    void va_copy(va_list dst, va_list src);
