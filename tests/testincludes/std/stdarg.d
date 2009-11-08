
/*
 * Placed in public domain.
 * Written by Hauke Duden and Walter Bright
 */

/* This is for use with variable argument lists with extern(D) linkage. */

module std.stdarg;

alias void* va_list;

template va_arg(T)
{
    T va_arg(inout va_list _argptr)
    {
	T arg = *cast(T*)_argptr;
	_argptr = _argptr + ((T.sizeof + int.sizeof - 1) & ~(int.sizeof - 1));
	return arg;
    }
}

