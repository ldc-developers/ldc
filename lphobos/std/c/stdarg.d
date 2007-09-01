
/**
 * C's &lt;stdarg.h&gt;
 * Authors: Hauke Duden and Walter Bright, Digital Mars, www.digitalmars.com
 * License: Public Domain
 * Macros:
 *	WIKI=Phobos/StdCStdarg
 */

/* This is for use with extern(C) variable argument lists. */

module std.c.stdarg;

alias void* va_list;

template va_start(T)
{
    void va_start(out va_list ap, inout T parmn)
    {
	ap = cast(va_list)(cast(void*)&parmn + ((T.sizeof + int.sizeof - 1) & ~(int.sizeof - 1)));
    }
}

template va_arg(T)
{
    T va_arg(inout va_list ap)
    {
	T arg = *cast(T*)ap;
	ap = cast(va_list)(cast(void*)ap + ((T.sizeof + int.sizeof - 1) & ~(int.sizeof - 1)));
	return arg;
    }
}

void va_end(va_list ap)
{

}

void va_copy(out va_list dest, va_list src)
{
    dest = src;
}
